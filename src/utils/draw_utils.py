import cv2

def putText_outline(
    img,
    text,
    org,
    fontFace,
    fontScale,
    color,
    thickness,
    outline_color=(255, 255, 255),
    outline_thickness=None,
):
    """
    Render text with an outline to improve readability against complex backgrounds.

    The function first draws the outline using a thicker stroke, followed by the
    foreground text using the specified color and thickness.

    Parameters
    ----------
    img : np.ndarray
        Image on which the text will be rendered.
    text : str
        Text string to draw.
    org : tuple[int, int]
        Bottom-left corner of the text in pixel coordinates.
    fontFace : int
        OpenCV font type (e.g., ``cv2.FONT_HERSHEY_SIMPLEX``).
    fontScale : float
        Font scale factor.
    color : tuple[int, int, int]
        Foreground text color in BGR format.
    thickness : int
        Thickness of the foreground text.
    outline_color : tuple[int, int, int], optional
        Outline color in BGR format. Defaults to white.
    outline_thickness : int, optional
        Thickness of the outline. If ``None``, it is automatically set to
        ``thickness + 4``.

    Returns
    -------
    None
        The input image is modified in place.
    """
    if outline_thickness is None:
        outline_thickness = thickness + 4

    cv2.putText(
        img, text, org, fontFace, fontScale,
        outline_color, outline_thickness, cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, fontFace, fontScale,
        color, thickness, cv2.LINE_AA
    )