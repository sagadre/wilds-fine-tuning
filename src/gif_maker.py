import imageio
import os
import glob


def write_gifs_scan(dump_dir):
    filelist = glob.glob(os.path.join(
        dump_dir, f"historical_warming_patterns_[0-9]*.png"))
    images = []
    for i in range(len(filelist)):
        images.append(imageio.imread(os.path.join(
            dump_dir, f"historical_warming_patterns_{i}.png")))

    out_path = os.path.join(
        dump_dir, f"diff_pred.gif")

    if len(images):
        print(f"saving to {out_path}")
        imageio.mimsave(out_path, images, duration=0.5)
        print(f"saved to {out_path}")

write_gifs_scan("./figs")