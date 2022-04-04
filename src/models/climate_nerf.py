from models.layers import Feedforward

class ClimateNerf(Feedforward):

    def __init__(self, layer_sizes=[3, 64, 64, 64, 64, 32, 1]):
        """_summary_

        Args:
            layer_sizes (list, optional): _description_. Defaults to [3, 64, 64, 64, 64, 32, 1].
            3 input channels representing latitude, longitude, and time.
                each is normalize so that they lie in the range of zero to 1
        """
        super().__init__(layer_sizes)
