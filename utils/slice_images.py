import os
import cv2
import time


###############################################################################
def slice_image(
    image_path,
    preject_name,
    out_dir_all_images,
    sliceHeight=416,
    sliceWidth=416,
    overlap=0.1,
    slice_sep="_",
    overwrite=False,
    out_ext=".png",
):
    if len(out_ext) == 0:
        im_ext = "." + image_path.split(".")[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    image = cv2.imread(image_path)  # , as_grey=False).astype(np.uint8)  # [::-1]
    print("image.shape:", image.shape)

    image_name = os.path.basename(image_path).split('.')[0]
    win_h, win_w = image.shape[:2]

    dx = int((1.0 - overlap) * sliceWidth)
    dy = int((1.0 - overlap) * sliceHeight)

    out_dir_image = os.path.join(out_dir_all_images, preject_name)


    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0 + sliceHeight > image.shape[0]:
                y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image.shape[1]:
                x = image.shape[1] - sliceWidth
            else:
                x = x0

            # extract image
            window_c = image[y : y + sliceHeight, x : x + sliceWidth]
            outpath = os.path.join(
                out_dir_image,
                image_name
                + slice_sep
                + str(y)
                + "_"
                + str(x)
                + "_"
                + str(sliceHeight)
                + "_"
                + str(sliceWidth)
                + "_"
                + str(win_w)
                + "_"
                + str(win_h)
                + im_ext,
            )
            if not os.path.exists(outpath):
                cv2.imwrite(outpath, window_c)
            elif overwrite:
                cv2.imwrite(outpath, window_c)
            else:
                print("outpath {} exists, skipping".format(outpath))

    print("Num slices:", n_ims, "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", image_path, time.time() - t0, "seconds")
    print(
        f"cliped results of {os.path.basename(image_path)} is saved at: {out_dir_image}"
    )
    return out_dir_image


# def slice_images(
#     images_dir,
#     outdir_slice_ims,
#     im_ext,
#     sliceHeight,
#     sliceWidth,
#     overlap,
#     slice_sep,
#     overwrite,
#     out_ext,
# ):
#     im_list = [z for z in os.listdir(images_dir) if z.endswith(im_ext)]
#     if not os.path.exists(outdir_slice_ims):
#         os.makedirs(outdir_slice_ims)  # , exist_ok=True)
#         # slice images
#     slice_images_dir_list = []
#     for i, im_name in enumerate(im_list):
#         im_path = os.path.join(images_dir, im_name)
#         im_tmp = cv2.imread(im_path)
#         h, w = im_tmp.shape[:2]
#         print(i, "/", len(im_list), im_name, "h, w =", h, w)

#         # tile data
#         out_name = im_name.split(".")[0].replace("_", "")
#         slice_images_dir = slice_image(
#             im_path,
#             out_name,
#             outdir_slice_ims,
#             sliceHeight=sliceHeight,
#             sliceWidth=sliceWidth,
#             overlap=overlap,
#             slice_sep=slice_sep,
#             overwrite=overwrite,
#             out_ext=out_ext,
#         )
#         slice_images_dir_list.append(slice_images_dir)

#     return slice_images_dir_list


if __name__ == "__main__":
    images_dir = r"E:\CS\GitHubClone\ultralytics\dataset\predict\init_images\IMG4521.JPG"
    outdir_slice_ims = r"E:\CS\GitHubClone\ultralytics\dataset\predict\slice_images"
    im_ext = ".JPG"
    slice_image(
        images_dir,
        outdir_slice_ims,
        im_ext,
        sliceHeight=1088,
        sliceWidth=1088,
        overlap=0.6,
        slice_sep="_",
        overwrite=False,
        out_ext=".png",
    )
