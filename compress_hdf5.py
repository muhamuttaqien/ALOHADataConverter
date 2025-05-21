import os
import time
import cv2
import h5py
import glob
import argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"]
DCAMERA_NAMES = ["dcam_high", "dcam_low"]

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return is_sim, qpos, qvel, effort, action, image_dict

def save_hdf5(file_info):
    episode_path, output_dir, is_compress, image_quality = file_info

    print (f'Load {episode_path}')
    try:
        is_sim, qpos, qvel, effort, action, image_dict = load_hdf5(episode_path)
        max_timesteps = len(action)

        dataset_name = os.path.basename(episode_path).split(".")[0]
        out_path         = os.path.join(output_dir, dataset_name + '.hdf5')

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
        }

        for cam_name in CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'] = []

        for i in range(max_timesteps):
            data_dict['/observations/qpos'].append(qpos[i])
            data_dict['/observations/qvel'].append(qvel[i])
            data_dict['/observations/effort'].append(effort[i])
            data_dict['/action'].append(action[i])
            for cam_name in CAMERA_NAMES:
                data_dict[f'/observations/images/{cam_name}'].append(image_dict[cam_name][i])

        if is_compress:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality] # image_quality: reccomend to be [20-50]
            compressed_len = []
            for cam_name in CAMERA_NAMES:
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in CAMERA_NAMES:
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')


        t0 = time.time()
        with h5py.File(out_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = is_sim
            root.attrs['compress'] = is_compress
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in CAMERA_NAMES:
                if is_compress:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            _ = obs.create_dataset('qpos', (max_timesteps, 14))
            _ = obs.create_dataset('qvel', (max_timesteps, 14))
            _ = obs.create_dataset('effort', (max_timesteps, 14))
            _ = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array

            if is_compress:
                _ = root.create_dataset('compress_len', (len(CAMERA_NAMES), max_timesteps))
                root['/compress_len'][...] = compressed_len
        print(f'Saving: {time.time() - t0:.1f} secs')
    except KeyboardInterrupt:
        print ("Stopped") 

def main(args):
    dataset_dir = args["dataset_dir"]
    output_dir  = args["output_dir"]
    is_compress  = args["compress"]
    image_quality = args["quality"]
    nproc       = args["nproc"]
    episodes = glob.glob(os.path.join(dataset_dir, '*.hdf5'), recursive=True)
    print (f'Load:{dataset_dir}')
    print (f'Size: {len(episodes)} episodes')
    
    pool = Pool(nproc)
    pool.map(save_hdf5, [(episode, output_dir, is_compress, image_quality) for episode in  episodes])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--output_dir', action='store', type=str, default="./data", help='Dataset dir.')
    parser.add_argument('--nproc', type=int, default=1, help='Dataset dir.', required=False)
    parser.add_argument('--compress', action='store_true', help='compress images', required=False)
    parser.add_argument('--quality', type=int, default=50, help='imencoding quality', required=False)
    main(vars(parser.parse_args()))  
