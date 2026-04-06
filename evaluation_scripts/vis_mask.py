import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from skimage import measure  # For contour finding


def _draw_mask_contours(ax, mask, color, alpha=1.0, linewidth=1):
    """
    绘制 Mask 的轮廓。

    Args:
        ax: Matplotlib Axes 对象。
        mask: 二值 Mask (H, W)。
        color: 轮廓颜色 (RGB 或 RGBA)。
        alpha: 轮廓透明度。
        linewidth: 轮廓线宽。
    """
    contours = measure.find_contours(mask, 0.5)  # 提取轮廓
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, alpha=alpha)


def visualize_image_and_masks_v1_enhanced(
    image: Image.Image,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    output_path: str = "visualization_v1_enhanced.png",
    pred_mask_color=(30, 144, 255),  # 预测 Mask 颜色：蓝色 (RGB)
    gt_mask_color=(50, 205, 50),  # 真实 Mask 颜色：绿色 (RGB)
    pred_mask_alpha=0.4,  # 预测 Mask 透明度
    gt_mask_alpha=0.3,  # 真实 Mask 透明度
    bbox: np.ndarray = None,  # 边界框: [x_min, y_min, x_max, y_max] (np.ndarray 或 list)
    point_coords: np.ndarray = None,  # 点坐标: [[x1, y1], [x2, y2], ...] (np.ndarray)
    point_labels: np.ndarray = None,  # 点标签: [1, 0, ...] (1 for positive, 0 for negative)
    display_text: str = None,  # 要显示的文本
    dpi=300,  # 输出图片DPI，影响清晰度
):
    """
    在第一版基础上，增强可视化功能：叠加 Mask，并额外绘制 BBox、点，在下方显示文本。
    """
    # 确保图像为RGB格式
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image) / 255.0  # 归一化到 0-1 范围，方便 Matplotlib 显示

    # 确保 Mask 为二值 (0 或 1) 且形状为 (H, W)
    pred_mask_bool = (pred_mask > 0).astype(bool).squeeze()
    gt_mask_bool = (gt_mask > 0).astype(bool).squeeze()

    if (
        pred_mask_bool.shape != image_np.shape[:2]
        or gt_mask_bool.shape != image_np.shape[:2]
    ):
        raise ValueError("Masks and image must have the same height and width.")

    img_height, img_width = image_np.shape[:2]

    text_area_ratio = 0.2 if display_text else 0
    fig_height_inches = img_height / dpi * (1 + text_area_ratio)
    fig_width_inches = img_width / dpi

    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, text_area_ratio], hspace=0)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_np)

    # 叠加真实 Mask
    gt_overlay_color_norm = np.array(gt_mask_color) / 255.0  # 颜色归一化到0-1
    gt_overlay_rgba = np.zeros(image_np.shape[:2] + (4,), dtype=np.float32)
    gt_overlay_rgba[gt_mask_bool, :3] = gt_overlay_color_norm
    gt_overlay_rgba[gt_mask_bool, 3] = gt_mask_alpha
    ax_img.imshow(gt_overlay_rgba)  # 显示RGBA图像
    _draw_mask_contours(ax_img, gt_mask_bool, color='green', alpha=1, linewidth=2) # 绘制 GT Mask 轮廓

    # 叠加预测 Mask (通常在真实 Mask 上方)
    pred_overlay_color_norm = np.array(pred_mask_color) / 255.0
    pred_overlay_rgba = np.zeros(image_np.shape[:2] + (4,), dtype=np.float32)
    pred_overlay_rgba[pred_mask_bool, :3] = pred_overlay_color_norm
    pred_overlay_rgba[pred_mask_bool, 3] = pred_mask_alpha
    ax_img.imshow(pred_overlay_rgba)
    _draw_mask_contours(ax_img, pred_mask_bool, color='blue', alpha=1, linewidth=1)  # 绘制 Pred Mask 轮廓

    # ==================== 绘制 BBox ====================
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax_img.add_patch(rect)

    # ==================== 绘制 Points ====================
    if point_coords is not None and point_labels is not None:
        pos_points = point_coords[point_labels == 1]
        neg_points = point_coords[point_labels == 0]

        if len(pos_points) > 0:
            ax_img.scatter(
                pos_points[:, 0],
                pos_points[:, 1],
                color="lime",
                marker="*",
                s=150,
                edgecolor="white",
                linewidth=1.25,
                label="Positive Point",
            )

        if len(neg_points) > 0:
            ax_img.scatter(
                neg_points[:, 0],
                neg_points[:, 1],
                color="darkred",
                marker="x",
                s=150,
                edgecolor="white",
                linewidth=1.25,
                label="Negative Point",
            )

        ax_img.set_axis_off()
        ax_img.set_xticklabels([])
        ax_img.set_yticklabels([])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.margins(0, 0)

    # ==================== 绘制文本区域 ====================
    if display_text:
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.set_facecolor("white")
        ax_text.text(
            0.02,
            0.5,
            display_text,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=10,
            wrap=True,
            transform=ax_text.transAxes,
        )
        ax_text.set_axis_off()

    plt.tight_layout(pad=0)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path, pad_inches=0)
    plt.close(fig)
    # print(f"可视化结果已保存到：{output_path}")


# --- 示例用法 (与之前相同，用于测试) ---
if __name__ == "__main__":
    # 1. 创建一个示例 PIL Image
    img_size = (256, 256)
    image = Image.fromarray(
        np.uint8(
            np.linspace(0, 255, img_size[0] * img_size[1]).reshape(
                img_size[0], img_size[1]
            )[:, :, np.newaxis].repeat(3, axis=2)
        )
    )
    image = image.resize(img_size)

    # 2. 创建示例 pred_mask 和 gt_mask
    y, x = np.ogrid[: img_size[0], : img_size[1]]
    center_y, center_x = img_size[0] // 2, img_size[1] // 2
    radius_gt = 60
    gt_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius_gt**2).astype(
        np.uint8
    )

    radius_pred = 65
    offset_x, offset_y = 10, -5
    pred_mask = ((x - (center_x + offset_x)) ** 2 + (y - (center_y + offset_y)) ** 2 <= radius_pred**2).astype(
        np.uint8
    )
    noise = np.random.rand(*img_size) > 0.95
    pred_mask = np.logical_or(pred_mask, noise).astype(np.uint8)

    # 3. 示例 BBox, Points, Text
    example_bbox = np.array([center_x - 70, center_y - 70, center_x + 70, center_y + 70])

    example_points = np.array(
        [[center_x + 5, center_y + 5], [center_x - 60, center_y - 60], [center_x + 100, center_y + 100]]
    )
    example_point_labels = np.array([1, 1, 0])

    example_text = ("test")

    # --- 调用增强后的可视化函数 ---
    print("\n--- 增强型可视化示例 ---")
    visualize_image_and_masks_v1_enhanced(
        image,
        pred_mask,
        gt_mask,
        output_path="output_enhanced_vis.png",
        pred_mask_alpha=0.5,
        gt_mask_alpha=0.3,
        bbox=example_bbox,
        point_coords=example_points,
        point_labels=example_point_labels,
        display_text=example_text,
    )

    print("\n脚本执行完毕。")