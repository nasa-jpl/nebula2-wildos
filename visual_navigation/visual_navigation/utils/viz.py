from typing import Optional, Dict, Tuple

import cv2
import numpy as np

def draw_point(
    img: np.ndarray, pt: np.ndarray, color: tuple, radius: int = 5
):
    """
    Draw a point on the image.

    :param img: The image to draw on.
    :param pt: np.ndarray of shape (2,) with (y,x) coordinates.
    :param color: Color for the point (r,g,b) in [0,255].
    :param radius: Radius of the point circle.
    """
    pt = tuple(np.round(pt[::-1]).astype(int))  # (x,y)
    cv2.circle(img, pt, radius, color, -1)


def draw_text(
    img: np.ndarray, pt: np.ndarray, text: str, color: tuple = (0, 0, 0)
):
    """
    Draw text on the image.

    :param img: The image to draw on.
    :param pt: np.ndarray of shape (2,) with (y,x) coordinates.
    :param text: The text string to draw.
    :param color: Color for the text (r,g,b) in [0,255].
    """
    pt = tuple(np.round(pt[::-1]).astype(int))  # (x,y)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 1, cv2.LINE_AA)

def draw_path(
    img: np.ndarray,
    path: np.ndarray,
    color: tuple,
    thickness: int = 2
):
    """
    Draw a path on the image.

    :param img: The image to draw on.
    :param path: np.ndarray of shape (N, 2) with (y,x) coordinates.
    :param color: Color for the path (r,g,b) in [0,255].
    :param thickness: Thickness of the path line.
    """
    for i in range(len(path) - 1):
        pt1 = tuple(np.round(path[i, ::-1]).astype(int))   # (x,y)
        pt2 = tuple(np.round(path[i+1, ::-1]).astype(int))
        cv2.line(img, pt1, pt2, color, thickness)

def pad_image(
    img: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Pad image with specified number of pixels on each side.

    :param img: The image to pad.
    :param top: Number of pixels to pad on the top.
    :param bottom: Number of pixels to pad on the bottom.
    :param left: Number of pixels to pad on the left.
    :param right: Number of pixels to pad on the right.
    :param color: Padding color (r,g,b) in [0,255].
    """
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

def add_title(
    img: np.ndarray, 
    title: str,
    font_scale: float = 0.7,
    thickness: int = 1,
    color: tuple = (0,0,0)
) -> np.ndarray:
    """
    Add title text above the image.

    :param img: The image to add title to.
    :param title: The title text.
    :param font_scale: Font scale for the title text.
    :param thickness: Thickness of the title text.
    :param color: Color for the title text (r,g,b) in [0,255].
    """
    (text_w, text_h), _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    out = pad_image(img, top=text_h + 10, color=(255,255,255))
    x = (out.shape[1] - text_w) // 2
    y = text_h + 5
    cv2.putText(
        out, title, (x,y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
    )
    return out

def make_subplot_grid(
    images_titles: Dict[Tuple[int, int], Tuple[np.ndarray, Optional[str]]],
    grid_shape: Tuple[int, int],
    pad: int = 10,
    title_font_scale: float = 0.7,
) -> np.ndarray:
    """
    Create a grid of images with titles.

    :param images_titles: Dictionary mapping (row, col) to (image, title) tuples. 
        Use None for no title.
    :param grid_shape: Tuple (rows, cols) specifying the grid shape.
    :param pad: Number of pixels to pad between images.
    """
    rows, cols = grid_shape

    # Prepare images and titles in grid format
    images = [[None for _ in range(cols)] for _ in range(rows)]
    titles = [[None for _ in range(cols)] for _ in range(rows)]
    for (r, c), (img, title) in images_titles.items():
        images[r][c] = img
        titles[r][c] = title

    assert len(images) == rows and all(len(row) == cols for row in images), \
        "Images must match the specified grid shape."
    assert len(titles) == rows and all(len(row) == cols for row in titles), \
        "Titles must match the specified grid shape."

    h_max, w_max = 0, 0
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            if images[r][c] is None:
                # white placeholder
                images[r][c] = np.ones((100, 100, 3), dtype=np.uint8) * 255
            if titles[r][c] is None:
                img = images[r][c]
            else:
                img = add_title(images[r][c], titles[r][c], font_scale=title_font_scale)
            row_imgs.append(img)
            h_max = max(h_max, img.shape[0])
            w_max = max(w_max, img.shape[1])
        processed.append(row_imgs)

    # Pad all images to same size
    for r in range(rows):
        for c in range(cols):
            img = processed[r][c]
            dh = h_max - img.shape[0]
            dw = w_max - img.shape[1]
            processed[r][c] = pad_image(img, bottom=dh, right=dw)

    # Concatenate into grid
    row_strips = [cv2.hconcat([pad_image(img, left=pad, right=pad) for img in row]) for row in processed]
    grid = cv2.vconcat([pad_image(strip, top=pad, bottom=pad) for strip in row_strips])

    return grid

def show_mask(
    base_img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
):
    """
    Show the binary mask overlaid on the base image.

    :param base_img: The base image (H, W, 3) in RGB format.
    :param mask: The binary mask (H, W) with values 0 or 1.
    :param alpha: The alpha blending factor for the mask overlay.
    """
    H, W, C = base_img.shape
    GREEN_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    green_layer = np.full((H, W, C), GREEN_COLOR, dtype=np.uint8)

    blended_img = cv2.addWeighted(base_img, alpha, green_layer, 1 - alpha, 0)
    valid_mask_2d = mask > 0
    valid_mask_3d = np.stack([valid_mask_2d] * C, axis=-1)

    fin_img = base_img.copy()
    fin_img[valid_mask_3d] = blended_img[valid_mask_3d]
    bbox = cv2.boundingRect(mask.astype(np.uint8))
    x,y,w,h = bbox
    bbox = np.array([x,y,x+w,y+h])
    cv2.rectangle(fin_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 6)

    return fin_img

def overlay_heatmap(
    base_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    vmin: float = 0,
    vmax: float = 1
) -> np.ndarray:
    """
    Overlay a heatmap on top of a base image.

    :param base_img: The base image (H, W, 3) in RGB format.
    :param heatmap: The heatmap to overlay (H, W) with values in [vmin, vmax].
    :param alpha: The alpha blending factor for the heatmap.
    :param vmin: Minimum value for heatmap normalization.
    :param vmax: Maximum value for heatmap normalization.
    """
    heatmap = np.clip((heatmap - vmin) / (vmax - vmin), 0, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(base_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

def make_colorbar(
    height: int,
    width: int,
    vmin: float = 0,
    vmax: float = 1,
    cmap=cv2.COLORMAP_JET,
    num_ticks: int = 10,
    font_scale: float = 0.5,
    pad: int = 5
) -> np.ndarray:
    """
    Create a vertical colorbar image.

    :param height: Height of the colorbar image.
    :param width: Width of the colorbar image.
    :param vmin: Minimum value for color mapping.
    :param vmax: Maximum value for color mapping.
    :param cmap: OpenCV colormap to use.
    """
    gradient = np.linspace(vmax, vmin, height).astype(np.float32)  # top-bottom
    gradient = np.tile(gradient[:, None], (1, width))
    gradient = np.uint8(255 * gradient)

    colorbar = cv2.applyColorMap(gradient, cmap)
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)

    # Add padding for labels
    colorbar = pad_image(colorbar, top=pad, bottom=pad, left=pad, right=pad, color=(255,255,255))
    cv2.rectangle(
        colorbar, (pad,pad), (colorbar.shape[1]-1-pad, colorbar.shape[0]-1-pad), (0,0,0), 1
    )
    # Add tick marks and labels
    for i in range(num_ticks + 1):
        y = pad + int(i * (height - 1) / num_ticks)
        cv2.line(colorbar, (pad-5, y), (pad, y), (0,0,0), 1)
        value = vmax - i * (vmax - vmin) / num_ticks
        label = f"{value:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(
            colorbar, label, (pad - text_w - 5, y + text_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA
        )

    return colorbar

def make_histogram(
    data: np.ndarray,
    bins: np.ndarray,
    img_shape: Tuple[int, int],
    bar_color: tuple = (0, 0, 255),
    bg_color: tuple = (255, 255, 255),
    pad: int = 10
) -> np.ndarray:
    """
    Create a histogram image from the input data.

    :param data: The data for each bin. (N,) array.
    :param bins: The bin edges for the histogram. (N+1,) array.
    :param img_shape: The shape of the output image.
    :param bar_color: The color of the histogram bars.
    :param bg_color: The background color of the image.
    :param pad: The padding to apply around the histogram.
    :return: The histogram image.
    """
    hist_img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) \
        * np.array(bg_color, dtype=np.uint8)
    
    cv2.rectangle(
        hist_img, (pad, pad), (hist_img.shape[1]-pad-1, hist_img.shape[0]-pad-1), (0,0,0), 1
    )
    # Add y ticks and labels
    num_ticks_y = 10
    max_data = np.max(data) if np.max(data) > 0 else 1.0
    for i in range(num_ticks_y + 1):
        y = pad + int(i * (hist_img.shape[0] - 1 - 2*pad) / num_ticks_y)
        cv2.line(hist_img, (pad-5, y), (pad, y), (0,0,0), 1)
        value = max_data - i * max_data / num_ticks_y
        label = f"{value:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
        cv2.putText(
            hist_img, label, (pad - text_w - 5, y + text_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,0), 1, cv2.LINE_AA
        )

    hist_height = hist_img.shape[0] - 1 - 2*pad
    hist_width = hist_img.shape[1] - 1 - 2*pad
    bar_width = hist_width / len(data)

    for i in range(len(data)):
        x1 = pad + int(i * bar_width)
        x2 = pad + int((i + 1) * bar_width)
        h = int((data[i] / max_data) * hist_height)
        cv2.rectangle(
            hist_img,
            (x1, hist_img.shape[0]-pad-1),
            (x2, hist_img.shape[0]-pad-1 - h),
            bar_color,
            -1
        )
        cv2.rectangle(
            hist_img,
            (x1, hist_img.shape[0]-pad-1),
            (x2, hist_img.shape[0]-pad-1 - h),
            (0,0,0),
            1
        )
        
    # Add x ticks and labels
    for i in range(len(bins)):
        x = pad + int(i * bar_width)
        cv2.line(hist_img, (x, hist_img.shape[0]-pad), (x, hist_img.shape[0]-pad+5), (0,0,0), 1)
        label = f"{bins[i]:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.1, 1)
        cv2.putText(
            hist_img, label, (x + (int(bar_width)-text_w)//2, hist_img.shape[0]-pad+text_h+5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,0), 1, cv2.LINE_AA
        )

    return hist_img
