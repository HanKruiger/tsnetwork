import subprocess
import glob
import shutil
import os
import modules.user_input as user_input


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


# Save animation of images from 'snaps_folder' as a .mp4 in the snaps_folder's
# parent folder.
def save_animation(snaps_folder, description):
    # chdir to snaps folder
    with cd(snaps_folder):
        # Create directory for reformatted images if it did not exists yet.
        if not os.path.exists('./animation'):
            os.makedirs('./animation')
        else:
            if user_input.confirm('[animation] ' + snaps_folder + '/animation exists. Delete contents?'):
                for file in os.listdir('./animation'):
                    file_path = os.path.join('./animation/', file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)

        # Copy files to new directory
        for file in glob.glob('./*.jpg'):
            shutil.copy(file, './animation')

        # chdir to snaps_folder/animation
        with cd('./animation'):
            # Write subprocesses' stdout to this file, to clean up logging.
            DEVNULL = open(os.devnull, 'w')

            # Reformat images with ImageMagick
            print('[animation] Reformatting images...')
            subprocess.call('mogrify -gravity center -resize 512x512 -extent 512x512 *.jpg', shell=True, stdout=DEVNULL)
            print('[animation] ... Done!')

            # Render video with ffmpeg
            print('[animation] Rendering video...')
            subprocess.call("ffmpeg -framerate 30 -pattern_type glob -i 'tsne_snap_*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p -hide_banner -loglevel panic {0}.mp4".format(description), shell=True, stdout=DEVNULL)
            print('[animation] ... Done!')
        try:
            # Copy video out of animation folder and remove the animation folder.
            shutil.copy('./animation/{0}.mp4'.format(description), '..')
            shutil.rmtree('./animation')
        except Exception as e:
            print(e)
