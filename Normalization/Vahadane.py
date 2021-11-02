import spams
class LuminosityThresholdTissueLocator():

    def get_tissue_mask(I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        assert(I.dtype == np.uint8)

        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = L < luminosity_threshold

        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")

        return mask
class VahadaneStainExtractor():

    def get_stain_matrix(I, luminosity_threshold=0.8, regularizer=0.1):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        assert(I.dtype == np.uint8)

        # convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # do the dictionary learning
        dictionary = spams.trainDL(X=OD.T, K=2, lambda1=regularizer, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T

        # order H and E.
        # H on first row.
        if dictionary[0, 0] < dictionary[1, 0]: dictionary = dictionary[[1, 0], :]

        return dictionnary/ np.linalg.norm(dictionnary, axis=1)[:, None]

class StainNormalizer(object):

    def __init__(self, method):
            self.extractor = VahadaneStainExtractor

    def fit(self, target):
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        self.stain_matrix_target_RGB = convert_OD_to_RGB(self.stain_matrix_target)  # useful to visualize.

    def transform(self, I):
        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = self.get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)

    def get_concentrations(I, stain_matrix, regularizer=0.01):
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)
    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)
