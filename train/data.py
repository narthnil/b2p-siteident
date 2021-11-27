from torch.utils.data import Dataset


class BridgesData(Dataset):

    def __init__(self, tile_size):
        """
        Get the extent of the Rwanda dataset and tile based on tile_size.
        Make sure to exclude any regions that aren't in Rwanda.

        Maybe:
        - For each "channel", extract the tile (INPUTS)
        - Determine binary: bridge is needed in area (OUTPUTS)
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
