from typing import Dict

import numpy as np


def augment_pop_data(img: np.ndarray, index_pop: int, stats: Dict,
                     loc: float = 0, scale: float = 3) -> np.ndarray:
    """Augments population data.

    The population data is augmented by random integer values that are Gaussian
    distributed. Finally we clip the values by the minimum and maximum
    population values according to the statistics.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_pop-th channel contains the population data.
        index_pop (int): Index of the population channel.
        stats (Dict): Dictionary containing the minimum (`min`) and maximum
            (`max`) values for clipping the population values.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the population data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the
            population data. Default: 3.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            population channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_pop, "Index cannot be accessed"
    assert "min" in stats and "max" in stats, \
        "stats does not have min and max keys."

    height, width, _ = img.shape
    # augments with rounded samples of Gaussian distribution N(loc, scale)
    img[:, :, index_pop:index_pop + 1] += np.random.normal(
        loc=loc, scale=scale, size=(height, width, 1)).round()
    # clip all values by stats min and max values for population
    img[:, :, index_pop:index_pop + 1] = np.clip(
        img[:, :, index_pop:index_pop + 1], stats["min"][0], stats["max"][0])
    return img


def augment_terrain_data(img: np.ndarray, index_terrain: int, loc: float = 0,
                         scale: float = 2) -> np.ndarray:
    """Augments terrain data.

    We augment each *unique* terrain (elevation / slope) value by a randomly
    Gaussian distributed value. The same terrain value gets the same random
    number. We sort the unique terrain values and we sort the random numbers.
    Thus, smaller terrain values get a smaller change than larger terrain
    values. In this way we make sure that the relative order of terrain values
    will not be changed through the augmentation.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_terrain-th channel contains the terrain data.
        index_terrain (int): Index of the terrain channel.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the terrain data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the
            terrain data. Default: 2.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            terrain channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_terrain, "Index cannot be accessed"

    img_terrain = img[:, :, index_terrain:index_terrain + 1]
    img_terrain_new = np.array(img[:, :, index_terrain:index_terrain + 1])
    # sort the unique terrain data
    unique_elems = sorted(np.unique(img_terrain).tolist())
    # sort the random samples from a Gaussian distribution
    # N(loc, scale)
    rnd = sorted(np.random.normal(
        loc=loc, scale=scale, size=len(unique_elems)).round().tolist())
    # add random samples to the terrain data
    for i, elem in enumerate(unique_elems):
        img_terrain_new[np.where(img_terrain == elem)] += rnd[i]
    img[:, :, index_terrain:index_terrain + 1] = img_terrain_new
    return img


def augment_osm_img(img: np.ndarray, index_img: int, loc: float = 0,
                    scale: float = 3) -> np.ndarray:
    """Augments OSM image data.

    The OSM image data is augmented by random integer values that are Gaussian
    distributed. Finally we clip the values by the minimum and maximum
    image pixel values (0 and 255).

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_img-th channel contains the OSM image data.
        index_img (int): Index of the OSM image channel.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the OSM image data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the OSM
            image data. Default: 2.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            OSM image channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_img, "Index cannot be accessed"
    height, width, _ = img.shape
    # augment with rounded Gaussian distributed samples N(loc, scale)
    img[:, :, index_img:index_img + 3] += np.random.normal(
        loc=loc, scale=scale, size=(height, width, 3)).round()
    # clip all values by min=0 and max=255
    img[:, :, index_img:index_img + 3] = np.clip(
        img[:, :, index_img:index_img + 3], 0, 255)
    return img


def augment_binary_map(img: np.ndarray, index_map: int,
                       p_reject: float = 0.5) -> np.ndarray:
    """Augments binary map data.

    A binary map has only two values, 0 and 1. We augment these maps by setting
    values 1 to 0 with a probability given by `p_reject`.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_map-th channel contains the binary data.
        index_img (int): Index of the binary channel.
        p_reject (float, optional): Probability that the pixel 1 is not flipped
            to 0. Default: 0.5.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            binary map channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_map, "Index cannot be accessed"
    assert 0 <= p_reject <= 1, \
        "Expected probability `p_reject` to be within [0, 1]."

    height, width, _ = img.shape
    # randomly sample a number between 0 and 1 for each pixel
    p = np.random.rand(height, width, 1)
    img_binary = img[:, :, index_map:index_map + 1]
    # if p > p_reject and the pixel is 1, then set it to 0
    img_binary[np.where(np.logical_and(p > p_reject, img_binary > 0))] = 0
    img[:, :, index_map:index_map + 1] = img_binary
    return img
