"""Convert one raw recording session into the legacy training HDF5 format.

Reader's guide: ``prepare_data_session`` is the orchestration entry point. It
loads ultrasound timestamps and images, aligns optional keyboard or piano
signals, reconstructs Vicon marker trajectories, derives joint and arm
kinematics, then writes the merged pandas table. The smaller functions below
are grouped in that same order so readers can follow the data flow from raw
files to the final ``/df`` HDF5 dataset.

This module reflects the original research preprocessing pipeline. It is not
executed during model training, and changing its alignment or kinematic
conventions would create a different dataset.
"""

import os
import sys
sys.path.append('.')
import time
import glob
import argparse
import datetime
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
from midiutil import MIDIFile

# ---------------------------------------------------------------------------
# Session orchestration
# ---------------------------------------------------------------------------

def prepare_data_session(data_dir, output_dir, img_res, use_imgs=False, resize_imgs=True, right_handed=True, modified=True, print_joints=True, verbose=True, use_calib=False):
    """Load, align, derive, and serialize every stream for one recording."""
    
    # create dictionary to store statistics
    stat_dict = {}
    if data_dir[-1] != '/':
        data_dir += '/'
    stat_dict['subject'] = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_dir)))))
    stat_dict['category'] = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(data_dir))))
    stat_dict['enrollment'] = os.path.basename(os.path.dirname(os.path.dirname(data_dir)))
    stat_dict['session'] = os.path.basename(os.path.dirname(data_dir))
    midi_file = restored_sentence = None

    # prepare ultrasound data if exist
    if use_imgs and os.path.isfile(os.path.join(data_dir,'ultrasound_data.csv')):
        us_df, stat_dict = load_ultrasound_data(data_dir=data_dir, stat_dict=stat_dict, img_res=img_res, resize_imgs=resize_imgs, verbose=verbose, use_calib=use_calib)

    # prepare vicon data if exist
    if os.path.isdir(os.path.join(data_dir,'vicon')):
        vicon_df, stat_dict = load_vicon_data(data_dir=data_dir, stat_dict=stat_dict, right_handed=right_handed, modified=modified, verbose=verbose)

        # join images with vicon measurements using timestamp
        if use_imgs:
            if verbose:
                print('Merging ultrasound images with Vicon data...')
            data_df = pd.merge_asof(us_df.sort_values('datetime'), vicon_df, on='datetime', direction='nearest', tolerance=pd.Timedelta("100ms"))
            samples_lost = 0
            if len(data_df.dropna()) < len(us_df):
                samples_lost = len(us_df) - len(data_df.dropna())
                if samples_lost > 0 and verbose:
                    print(f'Lost {samples_lost} samples when merging images with Vicon data')
                data_df = data_df.dropna()
            stat_dict['lost_in_merge'] = samples_lost
        else:
            data_df = vicon_df

    # prepare keyboard data if found
    if stat_dict['category'] == 'typing' and 'calib' not in stat_dict['session']:
        keys_df, restored_sentence, stat_dict = load_keyboard_data(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

        # join keys with the reset of the measurements
        if verbose:
            print('Merging data with keyboard keys...')
        data_df_size = len(data_df)
        data_df = pd.merge_asof(data_df.sort_values('datetime'), keys_df, on='datetime', direction='nearest', tolerance=pd.Timedelta("10ms"))
        samples_lost = 0
        if len(data_df.dropna()) < data_df_size:
            samples_lost = data_df_size - len(data_df.dropna())
            if samples_lost > 0 and verbose:
                print(f'Lost {samples_lost} samples when merging images with keyboard keys.')
            data_df = data_df.dropna()
        stat_dict['lost_in_merge_kbkeys'] = samples_lost
    
    if stat_dict['category'] == 'playing' and 'calib' not in stat_dict['session']:
        notes_df, stat_dict, midi_file = load_piano_data(data_dir=data_dir, vicon_df=vicon_df, stat_dict=stat_dict, verbose=verbose)

        # join keys with the reset of the measurements
        if verbose:
            print('Merging data with piano notes...')
        data_df_size = len(data_df)
        data_df = pd.merge_asof(data_df.sort_values('datetime'), notes_df, on='datetime', direction='nearest', tolerance=pd.Timedelta("10ms"))
        samples_lost = 0
        if len(data_df.dropna()) < data_df_size:
            samples_lost = data_df_size - len(data_df.dropna())
            if samples_lost > 0 and verbose:
                print(f'Lost {samples_lost} samples when merging images with piano notes.')
            data_df = data_df.dropna()
        stat_dict['lost_in_merge_pnotes'] = samples_lost

    # check if output folder exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # plot joint angles for visualization
    if print_joints:
        plot_joint_angles_graphs(data_df=data_df, output_dir=output_dir, data_dir=data_dir, verbose=verbose)

    # save as h5
    save_processed_h5(data_df=data_df, output_dir=output_dir, data_dir=data_dir, midi_file=midi_file, restored_sentence=restored_sentence, verbose=verbose)

    return stat_dict

# ---------------------------------------------------------------------------
# Source readers: ultrasound, Vicon, keyboard, and piano streams
# ---------------------------------------------------------------------------

def load_ultrasound_data(data_dir, stat_dict, img_res, resize_imgs=True, verbose=True, use_calib=False):
    """Load timestamped ultrasound frames and normalize their representation."""

    if verbose:
        print('Loading ultrasound images...')

    # load ultrasound data csv
    us_df = pd.read_csv(os.path.join(data_dir,'ultrasound_data.csv'))
    us_df['time'] = us_df['time'].round(0).astype(int)

    # load all ultrasound images
    us_images_list = glob.glob(os.path.join(data_dir, "images", "*.png"))
    us_imgs_df = pd.DataFrame(data={'path':us_images_list})
    us_imgs_df['time'] = us_imgs_df['path'].apply(lambda x: int(os.path.basename(x).split('.png')[0]))
    us_imgs_df['img'] = us_imgs_df['path'].apply(lambda x: Image.open(x).convert('RGB'))
    # Previous: converted to grayscale 'L', now using RGB to preserve 3-channel color information 

    # crop and resize images
    if resize_imgs:
        img_size = us_imgs_df['img'].iloc[0].size
        img_actual_size = min(img_size)
        img_offset = abs(img_size[0] - img_size[1])
        if img_offset > 0:
            us_imgs_df['img'] = us_imgs_df['img'].apply(lambda x: x.crop((img_offset // 2, 0, img_actual_size + (img_offset // 2), img_actual_size)))
        us_imgs_df['img'] = us_imgs_df['img'].apply(lambda x: x.resize((img_res, img_res)))

    # load segmentation masks if they exist
    masks_dir = os.path.join(data_dir, "masks")
    masks_available = False
    if os.path.isdir(masks_dir):
        if verbose:
            print('Loading segmentation masks...')

        # load all mask images
        mask_images_list = glob.glob(os.path.join(masks_dir, "*.png"))
        masks_df = pd.DataFrame(data={'mask_path': mask_images_list})
        masks_df['time'] = masks_df['mask_path'].apply(lambda x: int(os.path.basename(x).split('.png')[0]))
        masks_df['mask'] = masks_df['mask_path'].apply(lambda x: np.array(Image.open(x)))

        # crop and resize masks using nearest neighbor to preserve label values
        if resize_imgs:
            if img_offset > 0:
                masks_df['mask'] = masks_df['mask'].apply(lambda x: Image.fromarray(x).crop((img_offset // 2, 0, img_actual_size + (img_offset // 2), img_actual_size)))
                masks_df['mask'] = masks_df['mask'].apply(lambda x: np.array(x.resize((img_res, img_res), Image.NEAREST)))
            else:
                masks_df['mask'] = masks_df['mask'].apply(lambda x: np.array(Image.fromarray(x).resize((img_res, img_res), Image.NEAREST)))

        # merge masks with images dataframe
        us_imgs_df = us_imgs_df.merge(masks_df[['time', 'mask']], how='left', on='time')
        masks_available = True

        if verbose:
            initial_masks = us_imgs_df['mask'].notna().sum()
            print(f'{initial_masks} segmentation masks loaded initially.')
    else:
        if verbose:
            print('No masks directory found, skipping segmentation masks.')

    # merge meta-data csv with images
    us_df_merged = us_df.merge(us_imgs_df, how='inner', on='time')
    lost_us_samples = 0
    if len(us_df_merged) < max(len(us_df),len(us_imgs_df)):
        lost_us_samples = max(len(us_df),len(us_imgs_df)) - len(us_df_merged)
        if verbose:
            print(f'Lost {lost_us_samples} samples when merging images with meta-data')
    stat_dict['Lost_us_samples'] = lost_us_samples

    # Update mask statistics after final merge to reflect what's actually in the H5 file
    if masks_available:
        masks_loaded = us_df_merged['mask'].notna().sum()
        masks_missing = us_df_merged['mask'].isna().sum()

        if verbose and lost_us_samples > 0:
            print(f'Final count after merge: {masks_loaded} masks in H5 file.')

        stat_dict['masks_loaded'] = masks_loaded
        stat_dict['masks_missing'] = masks_missing
    else:
        stat_dict['masks_loaded'] = 0
        stat_dict['masks_missing'] = 0

    # # filter timestamps to cope with wifi delay - dismiss first frame
    # us_df_merged['rtime_norm'] = us_df_merged['rtime']- us_df_merged['rtime'].iloc[1]
    # us_df_merged['time_norm'] = (us_df_merged['time']- us_df_merged['time'].iloc[1]) / 1e9
    # us_df_merged['time_diff'] = us_df_merged['time_norm'] - us_df_merged['rtime_norm']
    # time_mean_diff = us_df_merged.time_diff.iloc[1:].mean()
    # stat_dict['us_time_mean_diff'] = time_mean_diff
    # us_df_merged['time_diff'].iloc[0] = time_mean_diff
    # us_df_merged['rtime'] = us_df_merged['rtime'] + us_df_merged['time_diff'] - time_mean_diff

    # compute average difference in time from world clock
    time_diff = us_df_merged.rtime - (us_df_merged.time * 1e-9)
    stat_dict['us_time_diff_mean'] = time_diff.mean()
    stat_dict['us_time_diff_std'] = time_diff.std()

    # load calibration data if asked
    if use_calib and 'time_stats.csv' in os.listdir(data_dir):
        calib_df = pd.read_csv(os.path.join(data_dir, 'time_stats.csv'))

        # get time coefficient
        time_coeff = calib_df.time_coeff.iloc[-1]
        stat_dict['us_calib_time_coeff_mean'] = time_coeff
        stat_dict['us_calib_time_coeff_std'] = calib_df.coeff_std.iloc[-1] 

    else: # compromise: use average difference in time for original clock
        time_coeff = time_diff.mean()

    # apply difference in time to original clock
    us_df_merged['time_adj'] = (us_df_merged.time * 1e-9) + time_coeff

    # convert timestamp to datetime format
    if 'time_adj' in us_df_merged.columns:
        us_df_merged['datetime'] = us_df_merged['time_adj'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    else:
        us_df_merged['datetime'] = us_df_merged['rtime'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    #us_df_merged = us_df_merged.drop(columns=['time','rtime','path','rtime_norm','time_norm','time_diff'])
    us_df_merged = us_df_merged.drop(columns=['path','time_adj'])
    #us_df_merged = us_df_merged.drop(columns=['time','rtime','path'])
    us_df_merged['datetime'] = pd.to_datetime(us_df_merged['datetime'])

    if verbose:
        print('{} ultrasound images loaded.'.format(len(us_df_merged)))
    stat_dict['us_images_loaded'] = len(us_df_merged)

    return us_df_merged, stat_dict

def load_vicon_data(data_dir, stat_dict, joints='all', right_handed=True, modified=True, verbose=True, c_joints=True, c_arm=True, c_ders=True):
    """Load marker trajectories and optionally derive joints, arm pose, and rates."""

    if verbose:
        print('Loading Vicon measurements...')

    # load vicon data csv
    hand_df, stat_dict = load_vicon_csv(data_dir=data_dir, stat_dict=stat_dict, right_handed=right_handed, modified=modified, verbose=verbose)

    # convert timestamp to datetime format
    hand_df = convert_time_to_datetime(hand_df)

    # convert measurements to numeric values and remove nans
    hand_df, link_names, _, stat_dict = convert_vicon_data_to_numeric(data_df=hand_df, stat_dict=stat_dict, verbose=verbose)
    
    # for each link, separate xyz into 3 separated columns
    #hand_df, link_names_separated = separate_link_columns(hand_df, link_names)

    # compute euclidean distances
    #distance_prev_df, distance_next_df = compute_euclidean_distances(hand_df, link_names_separated)
    
    # put NaN in cells with detached markers (locations with [0,0,0])
    #hand_df = replace_detached_markers_with_nan(hand_df, link_names_separated)

    # replace values that do not cope with the distance threshold with NaN
    #hand_df = remove_values_by_distance(df=hand_df, link_names_separated=link_names_separated, prevs_dist=distance_prev_df, nexts_dist=distance_next_df, movement_thresh=5, fill_val=np.nan)

    # interpolate values for each dimension
    #hand_df = interpolate_missing_markers(hand_df=hand_df, link_names_separated=link_names_separated, method='linear')

    # merge separated link names back to 3-d positions
    #hand_df = merge_separated_links(hand_df, link_names, link_names_separated)

    # fixing unwanted markers swap
    #hand_df = fixing_markers_swap(hand_df)

    # translate measurements to be w.r.t. thumb1
    #hand_df = translate_measurements(hand_df=hand_df, link_names=link_names, new_origin='thumb1')

    # prepare joint angles to compute: {joint name: [origin link, point a link, point b link]}
    if c_joints:

        # compute normal vectors for the hand palm 
        hand_df = compute_normal_to_plane(df=hand_df, origin='thumb4', point_a='finger11', point_b='finger41', normal_name='palm_normal')
        hand_df = compute_normal_vector_from_points(df=hand_df, point_a='finger41', point_b='finger11', vector_name='palm_horizontal')

        # define links and vectors for joints
        angles = {'joint41':['finger41','thumb4','finger42', False, False], # fingers - first row
                'joint31':['finger31','thumb4','finger32', False, False],
                'joint21':['finger21','thumb4','finger22', False, False],
                'joint11':['finger11','thumb4','finger12', False, False],
                'joint42':['finger42','finger41','finger43', False, False], # fingers - second row
                'joint32':['finger32','finger31','finger33', False, False],
                'joint22':['finger22','finger21','finger23', False, False],
                'joint12':['finger12','finger11','finger13', False, False],
                'joint43':['finger43','finger42','finger44', False, False], # fingers - third row
                'joint33':['finger33','finger32','finger34', False, False],
                'joint23':['finger23','finger22','finger24', False, False],
                'joint13':['finger13','finger12','finger14', False, False],
                'jointt5':['thumb5','thumb6','palm_normal', True, True], # thumb - 2 joints
                'jointt6':['thumb6','thumb5','thumb7', False, False],
                'jointwr':['thumb2','thumb1','palm_normal', True, False], # wrist - 3 DOFs
                'jointwp':['thumb4','thumb3','palm_normal', True, False],
                'jointwy':['thumb4','thumb3','palm_horizontal', True, False],
                # accumulated joints for each finger
                'jointt4':['finger41','thumb4','finger43', False, False], # finger 4
                'jointt3':['finger31','thumb4','finger33', False, False], # finger 3
                'jointt2':['finger21','thumb4','finger23', False, False], # finger 2
                'jointt1':['finger11','thumb4','finger13', False, False], # finger 1
                'jointtt':['thumb5','thumb7','palm_normal', True, True] # thumb
        }

        # old wrist
        # 'wristy':['thumb4','thumb3','finger21', False]

        # compute joint parameters (angles)
        if verbose:
            print('Computing joint angles...')
        for joint_name, links in angles.items():
            hand_df, outliers_count = compute_joint_angle(df=hand_df, origin=links[0], point_a=links[1], point_b=links[2], use_vector=links[3], invert_joint=links[4], joint_name=joint_name, verbose=verbose)
            stat_dict[joint_name + '_outliers'] = outliers_count

    # compute data of arm (velocities and accelerations)
    if c_arm:
        hand_df = compute_arm_data(df=hand_df, verbose=verbose)

    # compute first and second order derivatives for joint angles
    if c_ders:
        if verbose:
            print('Computing joint angles derivatives...')
        for joint_name in angles.keys():
            hand_df = compute_derivative_for_joint(df=hand_df, joint_name=joint_name)
            hand_df = compute_derivative_for_joint(df=hand_df, joint_name=joint_name + '_der')

    # add meta-data (arm relative to keyboard position)
    if stat_dict['category'] == 'typing' or stat_dict['category'] == 'playing':
        hand_df, stat_dict = compute_arm_relative_to_device(df=hand_df, data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    if verbose:
        print('{} Vicon measurements loaded.'.format(len(hand_df)))
    return hand_df, stat_dict

def load_keyboard_data(data_dir, stat_dict, verbose=True):
    """Read keyboard events and align their timestamps with the recording clock."""

    if verbose:
        print('Loading keyboard data...')

    # load keyboard dataframe
    keys_df = pd.read_csv(os.path.join(data_dir, "qwerty_data.csv"))

    # dismiss first rows if starting from 1 (meaning user started pressing before recording was set)
    first_pressing_idx = keys_df[keys_df['event'] == 0].event.idxmin()
    if first_pressing_idx > 0:
        keys_df = keys_df.iloc[first_pressing_idx:]

    # make sure that there are only press and release events
    keys_df = keys_df[(keys_df['event'] == 0) | (keys_df['event'] == 1)]

    # prepare text file of the restored sentence
    pressed_keys_asciis = keys_df[keys_df['event'] == 1]['key'].values
    pressed_keys_asciis[np.where(pressed_keys_asciis == 20)[0]] = 32 # fix space ascii char
    restored_sentence = ''.join(chr(i) for i in pressed_keys_asciis) # convert array of ascii chars to visible text
    stat_dict['restored_sentence'] = restored_sentence

    # only if it's multi keys session
    if 'multi' in stat_dict['session']:

        # load keys classes dataframe
        if not os.path.isfile(os.path.join(data_dir, "keys_classes.csv")):
            print('ERROR: classes file is not exist for multi keys session!')
            sys.exit(0)
        keys_classes_df = pd.read_csv(os.path.join(data_dir, "keys_classes.csv"))
        keys_classes_dict = {x:y for x,y in zip(keys_classes_df['ascii'].tolist(), keys_classes_df['class'].tolist())}

        # make sure there are no more than five pressed keys
        selected_keys = sorted(keys_df.key.value_counts().iloc[0:5].index.tolist())
        keys_df = keys_df.loc[keys_df.key.isin(selected_keys)] # make sure to drop unwanted keys

        # add a column for class for multi key 
        keys_df['key_class'] = keys_df.key.apply(lambda x: keys_classes_dict[x])

    # create a one-hot vector 1s for pressed keys events and minus 1s for leaving keys events
    keys_df.key = keys_df.key.apply(lambda x: 26 if x == 20 else x) # put space at the end of the vector (index 26)
    keys_df.key = keys_df.key.apply(lambda x: x - 65 if x >= 65 else x) # put a-z to be indices 0-25
    keys_df['keys'] = keys_df.key.apply(lambda x: np.array([1 if i == x else 0 for i in range(27)])) # create a onehot per key
    keys_df['keys'] = keys_df.apply(lambda x: -1 * x['keys'] if x.event == 1 else x['keys'], axis=1)

    # accumulate the one-hot vectors to have only 1s if a key is being pressed at the same moment
    onehot_df = pd.DataFrame(np.vstack(keys_df['keys'].values), columns=[str(x) for x in range(27)]) # separate one-hot vectors into scalars
    onehot_df = onehot_df.cumsum() # accumulate values

    # notify indices of classes with accumulated value of more than one - currently don't fix automatically
    # legit_indices = np.argwhere(np.isnan(onehot_df[onehot_df > 1].idxmin().values)).squeeze()
    # nonlegit_indices = np.delete(np.arange(0,27), legit_indices)
    # if len(nonlegit_indices) > 0 and verbose:
    #     print(f'WARNING: Found {len(nonlegit_indices)} keyboard keys accumulated with more than one value!')
    # stat_dict['faulty_kb_keys'] = len(nonlegit_indices)

    # notify indices of classes that were found with non-zero classes at the end of the session
    nonlegit_indices = onehot_df.iloc[-1].loc[onehot_df.iloc[-1] != 0].index.tolist()
    if len(nonlegit_indices) > 0 and verbose:
        print(f'WARNING: Found {len(nonlegit_indices)} keyboard keys accumulated with more than one value! (free typing). Faulty keys:')
        print(nonlegit_indices)
    stat_dict['faulty_keys_ft_count'] = len(nonlegit_indices)
    stat_dict['faulty_keys_ft'] = nonlegit_indices

    # put onehots back as vectors
    keys_df['keys'] = list(onehot_df.values)

    # only if it's multi keys session
    if 'multi' in stat_dict['session']:

        # create a one-hot vector - 1s for pressed keys events and minus 1s for leaving keys events
        keys_df['keys_multi'] = keys_df.key_class.apply(lambda x: np.array([1 if i == x else 0 for i in range(5)])) # create a onehot per key
        keys_df['keys_multi'] = keys_df.apply(lambda x: -1 * x['keys_multi'] if x.event == 1 else x['keys_multi'], axis=1)

        # accumulate the one-hot vectors to have only 1s if a note is being pressed at the same moment
        onehot_df = pd.DataFrame(np.vstack(keys_df['keys_multi'].values), columns=[str(x) for x in range(5)]) # separate one-hot vectors into scalars
        onehot_df = onehot_df.cumsum() # accumulate values

        # notify indices of classes with accumulated value of more than one - currently don't fix automatically
        legit_indices = np.argwhere(np.isnan(onehot_df[onehot_df > 1].idxmin().values)).squeeze()
        nonlegit_indices = np.delete(np.arange(0,5), legit_indices)
        if len(nonlegit_indices) > 0 and verbose:
            print(f'WARNING: Found {len(nonlegit_indices)} keyboard keys accumulated with more than one value! (multi typing)')
        stat_dict['faulty_keys_mt'] = len(nonlegit_indices)

        # put onehots back as vectors
        keys_df['keys_multi'] = list(onehot_df.values)
        keys_df.drop(columns=['key_class'], inplace=True)

    # dismiss unwanted columns
    keys_df['datetime'] = keys_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    keys_df.drop(columns=['event','key','time'], inplace=True)

    # interpolate dataframe to make sure that the time window between frames is smaller enough
    # set boundaries for the new dataframe - larger in 10 seconds from each boundary
    new_earliest_time = keys_df.datetime.iloc[0] - pd.Timedelta('10s')
    new_latest_time = keys_df.datetime.iloc[-1] + pd.Timedelta('10s')
    # create new dataframe with the new time range
    new_keys_df = pd.date_range(start=new_earliest_time, end=new_latest_time, freq='5ms').to_frame(name='datetime')
    new_keys_df = new_keys_df.merge(keys_df, how='outer', on='datetime', sort=True)

    # pad values of onehot vectors with previous values
    new_keys_df['keys'].iloc[0] = np.zeros(len(keys_df['keys'].iloc[0]), dtype=np.int64)
    if 'multi' in stat_dict['session']:
        new_keys_df['keys_multi'].iloc[0] = np.zeros(5, dtype=np.int64)
    new_keys_df = new_keys_df.fillna(method='ffill')
    nan_values = new_keys_df['keys'].isnull().sum()
    if nan_values > 0 and verbose:
        print(f'WARNING: Found {nan_values} nan values for keys after merging!')
    stat_dict['kb_keys_nans'] = nan_values

    return new_keys_df, restored_sentence, stat_dict

def load_piano_data(data_dir, vicon_df, stat_dict, notes_num=88, notes_offset=21, verbose=True):
    """Read MIDI activity and align note events with Vicon frame timestamps."""

    if verbose:
        print('Loading piano data...')

    # load keyboard dataframe
    notes_df = pd.read_csv(os.path.join(data_dir, "midi_data.csv"))

    # dismiss first rows if starting with event 128 (meaning user started playing before recording was set)
    first_pressing_idx = notes_df[notes_df['event'] == 144].event.idxmin()
    if first_pressing_idx > 0:
        notes_df = notes_df.iloc[first_pressing_idx:]

    # dismiss pedal events or any other event which is not press or release
    notes_df = notes_df[(notes_df['event'] == 128) | (notes_df['event'] == 144)]

    # detect missing pressing or leaving events and remove them
    playing_start_time = notes_df.iloc[0].time - 1 # one second backwards for safety
    midi_notes_list = []
    cur_pressed_notes, faulty_note_events = {}, {}
    for index, row in notes_df.iterrows():

        if row.event == 128:
            if row.note in cur_pressed_notes:
                midi_notes_list.append([row.note, cur_pressed_notes[row.note][1] - playing_start_time, row.time - cur_pressed_notes[row.note][1], notes_df.loc[cur_pressed_notes[row.note][0]].velocity]) # pitch, time, duration, volume
                del cur_pressed_notes[row.note]
            else:
                faulty_note_events[index] = [row.note, row.time, row.event]
        else: # row.event == 144
            if row.note in cur_pressed_notes:
                faulty_note_events[index] = [row.note, cur_pressed_notes[row.note][1], cur_pressed_notes[row.note][2]]
            cur_pressed_notes[row.note] = [index, row.time, row.event]

    for row_note, note_data in cur_pressed_notes.items():
        if note_data[0] not in faulty_note_events:
            faulty_note_events[note_data[0]] = [row_note, note_data[1], note_data[2]]

    # report missing note pressing/leaving events
    if len(faulty_note_events) > 0:
        if verbose:
            print(f'WARNING: Found {len(faulty_note_events)} piano note events without related opening/closing event! (free playing). Faulty events:')
            for row_index, note_event in faulty_note_events.items():
                print(f'Row index: {row_index}, note: {note_event[0]}, event type: {note_event[2]}.')
    stat_dict['faulty_piano_notes_fp_count'] = len(faulty_note_events)
    stat_dict['faulty_piano_notes_fp'] = [val for key, val in faulty_note_events.items()]

    # remove events with missing completing events
    if len(faulty_note_events) > 0:
        notes_df.drop(index=list(faulty_note_events.keys()), inplace=True)

    # create a MIDI file and add tempo
    midi_file = MIDIFile(1)  # only 1 track
    bps = 100
    midi_file.addTempo(0, 0, 60 * bps)  # tempo: 50 per second

    # add midi notes
    midi_track_idx = 0
    channel = 0
    for midi_note_data in midi_notes_list:
        midi_file.addNote(midi_track_idx, channel, int(midi_note_data[0]), int(midi_note_data[1] * bps), int(midi_note_data[2] * bps), int(midi_note_data[3]))

    # get normalized note (putting the lowest note)
    notes_df['note_norm'] = notes_df.note.apply(lambda x: x - notes_offset) 

    # DECISION: keep notes played by left hand
    # remove unwanted notes that were pressed by left-hand (only for free playing)
    #if 'free' in stat_dict['session']:
    # piano_df, stat_dict = load_piano_positions(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)
    # notes_df, stat_dict = remove_piano_notes_from_left_hand(data_dir=data_dir, notes_df=notes_df, vicon_df=vicon_df, stat_dict=stat_dict, notes_num=notes_num, verbose=verbose)
    # piano_pos_columns = ['left_side1', 'left_side2', 'left_side3', 'right_side1', 'right_side2', 'right_side3']

    # create one-hot vector designed for multi-class classification
    # 1s for pressed notes events and minus 1s for leaving notes events
    notes_df['notes'] = notes_df.note_norm.apply(lambda x: np.array([1 if i == x else 0 for i in range(notes_num)])) # create a onehot per note
    notes_df['notes'] = notes_df.apply(lambda x: -1 * x['notes'] if x.event == 128 else x['notes'], axis=1)

    # accumulate the one-hot vectors to have only 1s if a note is being pressed at the same moment
    onehot_df = pd.DataFrame(np.vstack(notes_df['notes'].values), columns=[str(x) for x in range(notes_num)]) # separate one-hot vectors into scalars
    onehot_df = onehot_df.cumsum() # accumulate values

    # # notify indices of classes that were found with non-zero classes at the end of the session
    # nonlegit_indices = onehot_df.iloc[-1].loc[onehot_df.iloc[-1] != 0].index.tolist()
    # if len(nonlegit_indices) > 0 and verbose:
    #     print(f'WARNING: Found {len(nonlegit_indices)} piano notes accumulated with more than one value! (free playing). Faulty notes:')
    #     print(nonlegit_indices)
    # stat_dict['faulty_piano_notes_fp_count'] = len(nonlegit_indices)
    # stat_dict['faulty_piano_notes_fp'] = nonlegit_indices

    # put onehots back as vectors
    notes_df['notes'] = list(onehot_df.values)
    notes_df.drop(columns=['note_norm'], inplace=True)

    if 'multi' in stat_dict['session']:
        selected_notes = sorted(notes_df.note.value_counts().iloc[0:5].index.tolist())
        notes_df = notes_df.loc[notes_df.note.isin(selected_notes)] # make sure to drop unwanted notes
        notes_df['note_norm'] = notes_df.note.apply(lambda x: selected_notes.index(x)) # get normalized note
        # create a one-hot vector 1s for pressed notes events and minus 1s for leaving notes events
        notes_df['notes_multi'] = notes_df.note_norm.apply(lambda x: np.array([1 if i == x else 0 for i in range(5)])) # create a onehot per note
        notes_df['notes_multi'] = notes_df.apply(lambda x: -1 * x['notes_multi'] if x.event == 128 else x['notes_multi'], axis=1)

        # accumulate the one-hot vectors to have only 1s if a note is being pressed at the same moment
        onehot_df = pd.DataFrame(np.vstack(notes_df['notes_multi'].values), columns=[str(x) for x in range(5)]) # separate one-hot vectors into scalars
        onehot_df = onehot_df.cumsum() # accumulate values

        # notify indices of classes with accumulated value of more than one - currently don't fix automatically
        legit_indices = np.argwhere(np.isnan(onehot_df[onehot_df > 1].idxmin().values)).squeeze()
        nonlegit_indices = np.delete(np.arange(0,5), legit_indices)
        if len(nonlegit_indices) > 0 and verbose:
            print(f'WARNING: Found {len(nonlegit_indices)} piano notes accumulated with more than one value! (multi playing)')
        stat_dict['faulty_piano_notes_mp'] = len(nonlegit_indices)

        # put onehots back as vectors
        notes_df['notes_multi'] = list(onehot_df.values)
        notes_df.drop(columns=['note_norm'], inplace=True)
    elif 'single' in stat_dict['session']:

        # mark press and release events as 1 and 3
        notes_df['event'] = notes_df.event.apply(lambda x: 1 if x == 144 else 3)
        # iterate over the data to approximate pressing timestamp using midi velocity
        if len(notes_df) % 2 == 0:
            len_notes_df = len(notes_df)
        else:
            len_notes_df = len(notes_df) - 1
        for i in range(0, len_notes_df, 2):
            step_loc = 0.5 + ((notes_df.iloc[i+1].velocity - notes_df.iloc[i].velocity) / (127*2))
            pressing_time = notes_df.iloc[i].time + ((notes_df.iloc[i+1].time - notes_df.iloc[i].time) * step_loc)
            pressing_row = {'event'    : 2,
                            'time'     : pressing_time,
                            'datetime' : datetime.datetime.fromtimestamp(pressing_time)}
            notes_df = pd.concat([notes_df, pd.DataFrame([pressing_row])])

        # backfill values
        notes_df = notes_df.sort_values('datetime')
        notes_df = notes_df.fillna(method='ffill')

        # report faulty notes for single playing by counting unwanted number of events
        events_count = notes_df.event.value_counts().values
        stat_dict['piano_notes_faulty_sp'] = events_count.max() - events_count.min()

    # create datetime and dismiss irrelevant columns
    notes_df['datetime'] = notes_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    if 'single' not in stat_dict['session']:
        notes_df.drop(columns=['event'], inplace=True)
    if 'multi' in stat_dict['session']:
        notes_df.drop(columns=['velocity'], inplace=True)
    notes_df.drop(columns=['note','time'], inplace=True)

    # interpolate dataframe to make sure that the time window between frames is smaller enough
    # set boundaries for the new dataframe - larger in 10 seconds from each boundary (just in case...)
    new_earliest_time = notes_df.datetime.iloc[0] - pd.Timedelta('10s')
    new_latest_time = notes_df.datetime.iloc[-1] + pd.Timedelta('10s')
    # create new dataframe with the new time range
    new_notes_df = pd.date_range(start=new_earliest_time, end=new_latest_time, freq='5ms').to_frame(name='datetime')
    new_notes_df = new_notes_df.merge(notes_df, how='outer', on='datetime', sort=True)

    # merge with device position
    # new_notes_df = new_notes_df.merge(piano_df.drop(columns=['time']), how='outer', on='datetime', sort=True)

    # fill event and velocity with zeros for single playing
    if 'single' in stat_dict['session']:
        new_notes_df['event'].iloc[0] = 0
        new_notes_df['velocity'].iloc[0] = 0
        new_notes_df[['event','velocity']] = new_notes_df[['event','velocity']].fillna(method='ffill')

    # pad values with previous values
    new_notes_df['notes'].iloc[0] = np.zeros(notes_num, dtype=np.int64)
    # new_notes_df['dev_pos'].iloc[0] = piano_df['dev_pos'].iloc[0]

    if 'multi' in stat_dict['session']:
        new_notes_df['notes_multi'].iloc[0] = np.zeros(5, dtype=np.int64)
    if 'velocity' in new_notes_df.columns:
        new_notes_df['velocity'] = new_notes_df['velocity'].fillna(0)
    new_notes_df = new_notes_df.fillna(method='ffill')
    # pad values of piano positions for later visualization - no need to interpolate
    #if 'free' in stat_dict['session']:
    # new_notes_df[piano_pos_columns] = new_notes_df[piano_pos_columns].fillna(method='ffill').fillna(method='bfill')

    # report missing notes if any
    nan_values = new_notes_df['notes'].isnull().sum()
    if nan_values > 0 and verbose:
        print(f'WARNING: Found {nan_values} nan values for piano notes after merging!')
    stat_dict['piano_notes_nans_fp'] = nan_values
    if 'multi' in stat_dict['session']:
        nan_values = new_notes_df['notes_multi'].isnull().sum()
        if nan_values > 0 and verbose:
            print(f'WARNING: Found {nan_values} nan values for piano notes after merging! (multi playing)')
        stat_dict['piano_notes_nans_mp'] = nan_values

    return new_notes_df, stat_dict, midi_file

def load_piano_positions(data_dir, stat_dict, verbose=True):
    """Recover piano-key geometry from the recorded rigid-body measurements."""
    
    # create piano notes locations for removing left hand
    piano_df, stat_dict = load_piano_rigid_csv(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)

    # separate vicon columns into x,y,z for interpolation
    pos_columns = piano_df.drop(columns=['time', 'datetime']).columns.values.tolist()
    pos_columns_x = [x+'_x' for x in pos_columns]
    pos_columns_y = [x+'_y' for x in pos_columns]
    pos_columns_z = [x+'_z' for x in pos_columns]
    pos_columns_xyz = pos_columns_x + pos_columns_y + pos_columns_z
    piano_df[pos_columns_x] = piano_df[pos_columns].applymap(lambda x: x[0])
    piano_df[pos_columns_y] = piano_df[pos_columns].applymap(lambda x: x[1])
    piano_df[pos_columns_z] = piano_df[pos_columns].applymap(lambda x: x[2])

    # interpolate data to have data for timestamps of notes events
    piano_df = interpolate_missing_markers(hand_df=piano_df, link_names_separated=pos_columns_xyz, method='time', verbose=verbose)

    # compute mean location of keyboard
    piano_df['pos_x'] = piano_df[pos_columns_x].mean(axis=1)
    piano_df['pos_y'] = piano_df[pos_columns_y].mean(axis=1)
    piano_df['pos_z'] = piano_df[pos_columns_z].mean(axis=1)
    piano_df['dev_pos'] = piano_df.apply(lambda x: np.array([x.pos_x, x.pos_y, x.pos_z]), axis=1)

    # return separated columns back into 3D-arrays
    # for pos_column in pos_columns:
    #     piano_df[pos_column] = piano_df.apply(lambda x: np.array([x[pos_column+'_x'], x[pos_column+'_y'], x[pos_column+'_z']]), axis=1)
    # drop unwanted columns
    piano_df.drop(columns=pos_columns + pos_columns_xyz + ['pos_x', 'pos_y', 'pos_z'], inplace=True)

    return piano_df, stat_dict

def remove_piano_notes_from_left_hand(data_dir, notes_df, vicon_df, stat_dict, notes_num=88, verbose=True):
    """Discard note activations attributed to the non-recorded hand."""

    # define piano notes indices
    white_notes_idxs, black_notes_idxs = get_piano_notes_idxs()
    
    # create piano notes locations for removing left hand
    piano_df, stat_dict = load_piano_rigid_csv(data_dir=data_dir, stat_dict=stat_dict, verbose=verbose)
    piano_df, ws_us, ws_ls, bs_us, bs_ls = create_piano_notes_positions(piano_df=piano_df, white_notes_idxs=white_notes_idxs, black_notes_idxs=black_notes_idxs, verbose=verbose)

    # create dataframe with finger end-points and notes segments
    finger_columns = ['thumb6','finger13','finger23','finger33','finger43']
    fingers_df = vicon_df[['datetime'] + finger_columns]
    all_df = pd.merge(fingers_df, piano_df.drop(columns=['datetime']), left_index=True, right_index=True)

    # separate vicon columns into x,y,z for interpolation
    all_columns = all_df.drop(columns=['datetime']).columns.values.tolist()
    all_columns_x = [x+'_x' for x in all_columns]
    all_columns_y = [x+'_y' for x in all_columns]
    all_columns_z = [x+'_z' for x in all_columns]
    all_columns_xyz = all_columns_x + all_columns_y + all_columns_z
    all_df[all_columns_x] = all_df[all_columns].applymap(lambda x: x[0])
    all_df[all_columns_y] = all_df[all_columns].applymap(lambda x: x[1])
    all_df[all_columns_z] = all_df[all_columns].applymap(lambda x: x[2])

    # add notes events datetime data using merge 
    notes_columns = notes_df.columns
    notes_df['datetime'] = notes_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    all_df = all_df.merge(notes_df, how='outer', on='datetime', sort=True).drop(columns=notes_columns)

    # interpolate data to have data for timestamps of notes events
    all_df = interpolate_missing_markers(hand_df=all_df, link_names_separated=all_columns_xyz, method='time', verbose=verbose)

    # leave only timestamps related to notes events and return separated columns back into 3D-arrays
    all_df = all_df.merge(notes_df, how='inner', on='datetime', sort=True)
    for all_column in all_columns:
        all_df[all_column] = all_df.apply(lambda x: np.array([x[all_column+'_x'], x[all_column+'_y'], x[all_column+'_z']]), axis=1)
    all_df.drop(columns=all_columns_xyz, inplace=True)

    # prepare segments from points
    # white segments
    ws, bs = [], []
    for ws_upper, ws_lower in zip(ws_us, ws_ls):
        note_idx = ws_upper.split('_')[-1]
        all_df[f'w_{note_idx}'] = all_df.apply(lambda x: np.array([x[ws_upper], x[ws_lower]]), axis=1)
        ws.append(f'w_{note_idx}')
    # black segments
    for bs_upper, bs_lower in zip(bs_us, bs_ls):
        note_idx = bs_upper.split('_')[-1]
        all_df[f'b_{note_idx}'] = all_df.apply(lambda x: np.array([x[bs_upper], x[bs_lower]]), axis=1)
        bs.append(f'b_{note_idx}')
    all_df.drop(columns=ws_us+ws_ls+bs_us+bs_ls, inplace=True)

    # compute distance from each one of the fingertips to each of the segments
    for finger_column in finger_columns:
        all_df = compute_distance_from_point(df=all_df, segments=ws+bs, point_p=finger_column, white_notes_idxs=white_notes_idxs, black_notes_idxs=black_notes_idxs)

    # get_note_support_range
    srange_columns = ['srange_' + x for x in finger_columns]
    all_df['srange'] = all_df[srange_columns].apply(lambda x: np.array([np.concatenate(x).min(), np.concatenate(x).max()]), axis=1)
    all_df.drop(columns=srange_columns+ws+bs, inplace=True)

    all_df_bu = all_df.copy()

    # remove rows with notes that are out of the support range
    all_df['eligible'] = all_df.apply(lambda x: 1 if (x['note_norm'] >= x['srange'][0] and x['note_norm'] <= x['srange'][1]) else 0, axis=1)
    stat_dict['piano_left_hand_notes_num'] = len(all_df[all_df['eligible'] == 0]) // 2
    all_df = all_df[all_df['eligible'] == 1]
    stat_dict['piano_right_hand_notes_num'] = len(all_df) // 2
    if verbose:
        left_hand_notes_percentage = 100* stat_dict['piano_left_hand_notes_num']/(stat_dict['piano_right_hand_notes_num']+stat_dict['piano_left_hand_notes_num'])
        print('Removed {} notes that were played by left hand, being {:.2f} percent of all notes'.format(stat_dict['piano_left_hand_notes_num'], left_hand_notes_percentage))

    # return dataframe as notes dataframe
    return all_df.drop(columns=finger_columns+['srange','eligible']), stat_dict

# define piano notes indices for both black and whites
# ---------------------------------------------------------------------------
# Piano geometry helpers
# ---------------------------------------------------------------------------

def get_piano_notes_idxs():
    """Return the white- and black-key index sets used by geometry helpers."""

    white_notes_idxs = [0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 36, 38, 39, 41, 43, 44,
              46, 48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82, 84, 86, 87]
    black_notes_idxs = list(set(range(88)).difference(set(white_notes_idxs)))

    return white_notes_idxs, black_notes_idxs

def load_piano_rigid_csv(data_dir, stat_dict, verbose=True):
    """Load the piano rigid-body CSV associated with a session."""

    if verbose:
        print('Loading piano positions csv...')

    # load piano csv
    piano_csv_path = os.path.join(data_dir, 'vicon', 'piano_rigid_data.csv')
    piano_df = pd.read_csv(piano_csv_path)
    piano_df['datetime'] = piano_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # convert to numeric values
    column_names = piano_df.drop(columns=['time','datetime']).columns
    piano_df, stat_dict = convert_vicon_cells_to_numeric(data_df=piano_df, column_names=column_names, stat_dict=stat_dict, verbose=verbose, stat_label='piano_vicon_nan_count')

    return piano_df, stat_dict

# create piano notes locations for removing left hand
def create_piano_notes_positions(piano_df, white_notes_idxs, black_notes_idxs, verbose=True):
    """Map rigid-body measurements to estimated spatial positions for all keys."""

    if verbose:
        print('Creating piano notes positions...')

    # create a dynamic rectangle surrounding all notes
    left2_off, left3_off, right2_off, right3_off = 16.5, 18.19, 17.5, 17.7
    piano_df['ul_corner'] = piano_df.apply(lambda x: x['left_side2'] + (left2_off * (x['right_side2'] - x['left_side2'])/np.linalg.norm(x['right_side2'] - x['left_side2'])), axis=1)
    piano_df['ur_corner'] = piano_df.apply(lambda x: x['right_side2'] + (right2_off * (x['left_side2'] - x['right_side2'])/np.linalg.norm(x['left_side2'] - x['right_side2'])), axis=1)
    piano_df['ll_corner'] = piano_df.apply(lambda x: x['left_side3'] + (left3_off * (x['right_side3'] - x['left_side3'])/np.linalg.norm(x['right_side3'] - x['left_side3'])), axis=1)
    piano_df['lr_corner'] = piano_df.apply(lambda x: x['right_side3'] + (right3_off * (x['left_side3'] - x['right_side3'])/np.linalg.norm(x['left_side3'] - x['right_side3'])), axis=1)
    #piano_df.drop(columns=column_names, inplace=True)

    # divide the dynamic rectangle into 52 rectangles for white notes
    # create steps for the upper and lower part of each note
    piano_df['uw_step'] = piano_df.apply(lambda x: (x['ur_corner'] - x['ul_corner'])/len(white_notes_idxs), axis=1)
    piano_df['lw_step'] = piano_df.apply(lambda x: (x['lr_corner'] - x['ll_corner'])/len(white_notes_idxs), axis=1)
    white_segments_uppers, white_segments_lowers = [], []
    # iterate over all white notes and buile the rectangles
    for i, w_idx in enumerate(white_notes_idxs):

        if i > 0:
            piano_df[f'ul_w_{w_idx}'] = piano_df.apply(lambda x: x['ul_corner'] + (i * x['uw_step']), axis=1)
            #piano_df[f'ur_w_{w_idx}'] = piano_df.apply(lambda x: x['ul_corner'] + ((i+1) * x['uw_step']), axis=1)
            piano_df[f'll_w_{w_idx}'] = piano_df.apply(lambda x: x['ll_corner'] + (i * x['lw_step']), axis=1)
            #piano_df[f'lr_w_{w_idx}'] = piano_df.apply(lambda x: x['ll_corner'] + ((i+1) * x['lw_step']), axis=1)
            white_segments_uppers.append(f'ul_w_{w_idx}')
            white_segments_lowers.append(f'll_w_{w_idx}')
            
    piano_df[['ul_w_0','ul_w_88','ll_w_0','ll_w_88']] = piano_df[['ul_corner','ur_corner','ll_corner','lr_corner']]
    white_segments_uppers.append('ul_w_88')
    white_segments_lowers.append('ll_w_88')
    white_segments_uppers.insert(0,'ul_w_0')
    white_segments_lowers.insert(0,'ll_w_0')
    piano_df.drop(columns=['ul_corner','ur_corner','ll_corner','lr_corner'], inplace=True)
    #piano_df.drop(columns=['ul_corner','ur_corner','ll_corner','lr_corner','uw_step','lw_step'], inplace=True)

    # add 36 rectangles for black notes
    # create step from the upper and lower part of each note
    piano_df['b_step'] = piano_df.apply(lambda x: 93 * (x['ll_w_0'] - x['ul_w_0']) / np.linalg.norm(x['ll_w_0'] - x['ul_w_0']) ,axis=1)
    black_segments_uppers, black_segments_lowers = [], []
    # iterate over all black notes and define segments for them
    for j, b_idx in enumerate(black_notes_idxs):

        piano_df[f'ul_b_{b_idx}'] = piano_df.apply(lambda x: x[f'ul_w_{b_idx-1}'] + x['uw_step'] / 2, axis=1)
        piano_df[f'll_b_{b_idx}'] = piano_df.apply(lambda x: x[f'ul_b_{b_idx}'] + x['b_step'], axis=1)
        black_segments_uppers.append(f'ul_b_{b_idx}')
        black_segments_lowers.append(f'll_b_{b_idx}')
        if j == len(black_notes_idxs) - 1:
            piano_df[f'ul_b_{b_idx+1}'] = piano_df.apply(lambda x: x[f'ul_b_{b_idx}'] + x['uw_step'], axis=1)
            piano_df[f'll_b_{b_idx+1}'] = piano_df.apply(lambda x: x[f'ul_b_{b_idx+1}'] + x['b_step'], axis=1)
            black_segments_uppers.append(f'ul_b_{b_idx+1}')
            black_segments_lowers.append(f'll_b_{b_idx+1}')

    if 'time' in piano_df.columns:
        piano_df.drop(columns=['time'], inplace=True)

    return piano_df.drop(columns=['uw_step','lw_step','b_step']), white_segments_uppers, white_segments_lowers, black_segments_uppers, black_segments_lowers

# prepare dataframe to support distance computation for a group of segments from one point
def compute_distance_from_point(df, segments, point_p, white_notes_idxs, black_notes_idxs):
    """Compute distances from a tracked point to the modeled key segments."""

    # concatenate all segments of notes with  point 
    for segment in segments:
        df[f'{segment}_{point_p}'] = df.apply(lambda x: np.concatenate([x[segment], np.expand_dims(x[point_p], axis=0)]), axis=1)
    
    # compute distance for each of the point from the segment
    segments_with_points = [f'{x}_{point_p}' for x in segments]
    segments_distances = [f'{x}_distance' for x in segments]
    df[segments_distances] = df[segments_with_points].applymap(lambda x: compute_distance_to_segment(point_p=x[2], point_s1=x[0], point_s2=x[1]))

    # find closest note first
    df[f'closest_{point_p}_0'] = df[segments_distances].idxmin(axis=1).apply(lambda x: x.split('_')[:-1])
    
    # store its value and change it so we can find the second closest
    df[f'closest_{point_p}_value'] = df[segments_distances].min(axis=1)
    df[segments_distances + [f'closest_{point_p}_value']] = df[segments_distances + [f'closest_{point_p}_value']].apply(lambda x: x - x[f'closest_{point_p}_value'], axis=1)
    df[segments_distances] = df[segments_distances].applymap(lambda x: 1000 if x < 1e-10 else x)
    
    # find the second closest note
    df[f'closest_{point_p}_1'] =df[segments_distances].idxmin(axis=1).apply(lambda x: x.split('_')[:-1])
    
    # define note support range
    df[f'closest_{point_p}'] = df[[f'closest_{point_p}_0',f'closest_{point_p}_1']].apply(lambda x: sorted([int(x[f'closest_{point_p}_0'][1]),int(x[f'closest_{point_p}_1'][1])])[0], axis=1)
    df[f'srange_{point_p}'] = df[f'closest_{point_p}'].apply(lambda x: get_note_support_range(n_idx=x, ws=white_notes_idxs, bs=black_notes_idxs))

    # return dataframe with the new support range
    return df.drop(columns=segments_with_points + segments_distances + [f'closest_{point_p}_value',f'closest_{point_p}_0',f'closest_{point_p}_1',f'closest_{point_p}'])

# compute distance of point from segment
def compute_distance_to_segment(point_p, point_s1, point_s2):
    """Return the Euclidean distance from a point to a finite 3D segment."""

    # define vectors
    vec_s1p = point_p - point_s1
    vec_s1s2 = point_s2 - point_s1

    # compute projected point on segment
    projected_p = point_s1 + (vec_s1s2 * np.dot(vec_s1p, vec_s1s2) / np.linalg.norm(vec_s1s2)**2)

    # return distance to projected point
    return np.linalg.norm(point_p - projected_p)

# return notes support range by adding notes from the sides that might be related to the pressed finger
def get_note_support_range(n_idx, ws, bs):
    """Return neighboring key indices that geometrically support one note."""

    # if note type is white, make sure to add both adjacent black and white notes
    support_range = [n_idx-1, n_idx+1]
    if n_idx in ws:
        if n_idx-1 in bs:
            support_range[0] = n_idx-2
        if n_idx+1 in bs:
            support_range[1] = n_idx+2        

    return np.array(support_range)

# load vicon csv as dataframe
# ---------------------------------------------------------------------------
# Vicon cleaning, marker reconstruction, and derived kinematics
# ---------------------------------------------------------------------------

def load_vicon_csv(data_dir, stat_dict, right_handed, use_piano=False, modified=False, verbose=True):
    """Read the raw Vicon export and select the relevant marker columns."""

    # create csv path
    vicon_dir = os.path.join(data_dir, "vicon")
    if use_piano:
        csv_path = os.path.join(vicon_dir, "piano_rigid_data")
    elif right_handed:
        csv_path = os.path.join(vicon_dir, "right_hand_data")
    else:
        csv_path = os.path.join(vicon_dir, "left_hand_data")
    if modified:
        csv_path = csv_path + "_modified"
    csv_path = csv_path + ".csv"

    # load csv
    data_df = pd.read_csv(filepath_or_buffer=csv_path)
    if verbose:
        print('{} measurements detected.'.format(len(data_df)))
    stat_dict['vicon_len'] = len(data_df)

    return data_df, stat_dict

# convert timestamp to datetime format
def convert_time_to_datetime(hand_df):
    """Convert the Vicon time column to pandas datetime values."""

    hand_df['datetime'] = hand_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    hand_df['datetime'] = pd.to_datetime(hand_df['datetime'])
    return hand_df

# convert measurements to numeric values and remove nans
def convert_vicon_data_to_numeric(data_df, stat_dict, verbose=True):
    """Convert tracked-marker cells to numeric values and record conversion loss."""

    # get link names and other columns
    link_names, unwanted_columns = get_link_names(data_df=data_df)

    # convert cells of vicon dataframe to numeric
    data_df, stat_dict = convert_vicon_cells_to_numeric(data_df=data_df, column_names=link_names, stat_dict=stat_dict, verbose=verbose)

    return data_df.dropna(), link_names, unwanted_columns, stat_dict

# convert cells of vicon dataframe to numeric
def convert_vicon_cells_to_numeric(data_df, column_names, stat_dict, verbose=True, stat_label='vicon_nan_count'):
    """Convert selected Vicon columns to numeric values with NaN accounting."""

    # filter format "(x; y; z)" to [x, y, z]
    data_df[column_names] = data_df[column_names].applymap(lambda x: np.array(x.split('(')[-1].split(')')[0].replace(' ','').split(';'), dtype=np.float64))
    nan_values_count = data_df.isna().sum(axis=1).sum()
    if verbose:
        print('{} NaN values detected in Vicon dataframe.'.format(nan_values_count))
    stat_dict[stat_label] = nan_values_count

    return data_df.dropna(), stat_dict

# differentiate between link names and other columns and return the two lists
def get_link_names(data_df):
    """Extract unique tracked-link names from Vicon coordinate columns."""

    # get link names
    link_names = []
    for column_name in data_df.columns:
        if ('finger' in column_name) or ('probe' in column_name) or ('_side' in column_name) or\
           ('thumb' in column_name and '_der' not in column_name and '_dev' not in column_name):
            link_names.append(column_name)
    link_names = data_df[link_names].columns.values

    # get unwanted columns
    unwanted_columns = data_df.drop(columns=link_names).columns.values

    return link_names, unwanted_columns

# for each link, separate xyz into 3 separated columns
def separate_link_columns(hand_df, link_names):
    """Group each tracked link's coordinate columns for vectorized processing."""

    print('Separating links columns...')
    for link_name in link_names:
        hand_df[link_name + '_x'] = hand_df[link_name].apply(lambda x: x[0])
        hand_df[link_name + '_y'] = hand_df[link_name].apply(lambda x: x[1])
        hand_df[link_name + '_z'] = hand_df[link_name].apply(lambda x: x[2])
    hand_df = hand_df.drop(columns=link_names)
    # set separated joints list
    link_names_separated_lst = [[x+'_x',x+'_y',x+'_z'] for x in link_names.tolist()]
    link_names_separated = np.array(link_names_separated_lst).flatten()

    return hand_df, link_names_separated

# compute euclidean distances for each cell to the next and to the previous cells
def compute_euclidean_distances(hand_df, link_names_separated):
    """Compute frame-to-frame displacement for every tracked link."""

    print('Computing euclidean distances...')
    distance_next_df = (hand_df[link_names_separated].shift(1) - hand_df[link_names_separated]).applymap(lambda x: abs(x))
    distance_next_df = distance_next_df.fillna(0)
    distance_prev_df = (hand_df[link_names_separated].shift(-1) - hand_df[link_names_separated]).applymap(lambda x: abs(x))
    distance_prev_df = distance_prev_df.fillna(0)

    return distance_prev_df, distance_next_df

# put NaN in cells with detached markers (locations with [0,0,0])
def replace_detached_markers_with_nan(hand_df, link_names_separated):
    """Mark detached-marker sentinel values as missing measurements."""

    hand_df[link_names_separated] = hand_df[link_names_separated].applymap(lambda x: x if abs(x) > 1 else np.nan)
    print('Found approximately {} detached markers.'.format(int(hand_df.isnull().values.sum()/3)))

    return hand_df

# replace values that do not cope with the distance threshold with given value
def remove_values_by_distance(df, link_names_separated, prevs_dist, nexts_dist, movement_thresh=5, fill_val=0.0):
    """Suppress isolated marker jumps that exceed the movement threshold."""

    nexts_dist = nexts_dist.applymap(lambda x: x if x < movement_thresh else np.nan)
    prevs_dist = prevs_dist.applymap(lambda x: x if x < movement_thresh else np.nan)
    print('Found approximately {} misplaced markers.'.format(int((prevs_dist.isna() & nexts_dist.isna()).sum().sum()/3)))
    df[link_names_separated] = df[link_names_separated].mask((prevs_dist.isna() & nexts_dist.isna()), fill_val)

    return df

# fill values of missing markers
def interpolate_missing_markers(hand_df, link_names_separated, method, verbose=True):
    """Interpolate missing marker coordinates using the requested pandas method."""

    if verbose:
        print('Interpolating missing markers recordings...')
    
    if method == 'time':
        df_interpolated = hand_df.set_index('datetime').interpolate(method=method)
        return df_interpolated.reset_index()
    else:
        hand_df[link_names_separated] = hand_df[link_names_separated].interpolate(method=method)
        return hand_df

# merge separated link names back to 3-d positions
def merge_separated_links(hand_df, link_names, link_names_separated):
    """Merge processed per-link coordinates back into one frame table."""

    print("Merging separated links back...")
    for link_name in link_names:
        hand_df[link_name] = hand_df.apply(lambda x: np.array([x[link_name+'_x'],x[link_name+'_y'],x[link_name+'_z']]) , axis=1)
    return hand_df.drop(columns=link_names_separated)

# convert all measurements to be w.r.t. thumb1 as the main axis
def translate_measurements(hand_df, link_names, new_origin):
    """Express all tracked coordinates relative to a new origin marker."""

    hand_df['origin'] = hand_df[new_origin].copy()
    for link_name in link_names:
       hand_df[link_name] -= hand_df['origin']
    return hand_df.drop(columns=['origin'])

# compute data of arm (angles and velocities)
def compute_arm_data(df, verbose=True):
    """Derive arm orientation and related measurements from marker geometry."""

    # # The following code will compute euler angles of the plane composed of 'thumb3', 'thumb4' and 'thumb5'
    # # first, create 3 orthoginal unit vectors from the arm links (a, b and c)
    # self.train_df['arm_vec_a'] = self.train_df.apply(lambda x: (x['thumb4'] - x['thumb3']) / np.linalg.norm(x['thumb4'] - x['thumb3']), axis=1)
    # self.train_df['arm_vec_n'] = self.train_df.thumb5 - self.train_df.thumb4
    # self.train_df['arm_vec_b_full'] = self.train_df.apply(lambda x: x['arm_vec_n'] - (np.inner(x['arm_vec_a'],  x['arm_vec_n']) * x['arm_vec_a']), axis=1)
    # self.train_df['arm_vec_b'] = self.train_df['arm_vec_b_full'].apply(lambda x: x / np.linalg.norm(x))
    # self.train_df['arm_vec_c'] = self.train_df.apply(lambda x: np.cross(x['arm_vec_a'], x['arm_vec_b']), axis=1)

    # # rotation matrix is R = [a,b,c]. extract euler angles (phi, theta and psi) from cells
    # self.train_df['arm_phi'] = self.train_df.apply(lambda x: np.arctan2(-x['arm_vec_b'][0], x['arm_vec_a'][0]), axis=1)
    # self.train_df['arm_theta'] = self.train_df['arm_vec_c'].apply(lambda x: np.arcsin(x[0]))
    # self.train_df['arm_psi'] = self.train_df.apply(lambda x: np.arctan2(-x['arm_vec_c'][1], x['arm_vec_c'][2]), axis=1)
    # self.arm_angles_labels = ['arm_phi', 'arm_theta', 'arm_psi']

    # compute first and second order derivatives of arm locations
    if verbose:
        print('Computing arm links derivatives...')
    df = compute_derivative_for_link(df=df, link_name='thumb3')
    df = compute_derivative_for_link(df=df, link_name='thumb3_der')
    df = compute_derivative_for_link(df=df, link_name='thumb4')
    df = compute_derivative_for_link(df=df, link_name='thumb4_der')

    return df

def compute_derivative_for_link(df, link_name):
    """Add first and second temporal derivatives for one tracked link."""

    df[link_name + '_der'] = np.gradient(np.vstack(df[link_name].values), df.time.values, axis=0).tolist()
    df[link_name + '_der'] = df[link_name + '_der'].apply(lambda x: np.array(x))

    return df

def compute_derivative_for_joint(df, joint_name):
    """Add first and second temporal derivatives for one joint-angle signal."""

    df[joint_name + '_der'] = np.gradient(df[joint_name].values, df.time.values)
    return df

def compute_arm_relative_to_device(df, data_dir, stat_dict, verbose=True):
    """Express arm measurements in the ultrasound-device coordinate system."""

    # load keyboard measurements
    if stat_dict['category'] == 'playing':
        device_csv_path = os.path.join(data_dir, 'vicon', 'piano_rigid_data.csv')
    else: # stat_dict['category'] == 'playing'
        device_csv_path = os.path.join(data_dir, 'vicon', 'keyboard_rigid_data.csv')
    dev_df = pd.read_csv(device_csv_path)
    dev_df['datetime'] = dev_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # fix names of column names
    if 'arm3' in dev_df.columns:
        dev_df = dev_df.rename(columns={'arm3': 'right_side2'})
    if 'arm2' in dev_df.columns:
        dev_df = dev_df.rename(columns={'arm2': 'right_side3'})
    if 'arm1' in dev_df.columns:
        dev_df = dev_df.rename(columns={'arm1': 'left_side2'})

    # convert all links to single x y z
    dev_link_names = dev_df.columns.tolist()
    dev_link_names.remove('time')
    dev_link_names.remove('datetime')
    dev_link_names_x, dev_link_names_y, dev_link_names_z = [x+'_x' for x in dev_link_names], [x+'_y' for x in dev_link_names], [x+'_z' for x in dev_link_names]
    dev_df[dev_link_names_x] = dev_df[dev_link_names].applymap(lambda x: float(x.split(';')[0].split('(')[-1]))
    dev_df[dev_link_names_y] = dev_df[dev_link_names].applymap(lambda x: float(x.split('; ')[1]))
    dev_df[dev_link_names_z] = dev_df[dev_link_names].applymap(lambda x: float(x.split('; ')[-1].split(')')[0]))

    # interpolate missing values if any
    dev_link_names_xyz = dev_link_names_x + dev_link_names_y + dev_link_names_z
    dev_df[dev_link_names_xyz] = dev_df[dev_link_names_xyz].applymap(lambda x: x if abs(x) > 1e-3 else np.nan)
    if verbose:
        print('Found approximately {} detached device markers.'.format(int(dev_df[dev_link_names_xyz].isnull().values.sum()/3)))
    stat_dict['missing_dev_markers'] = int(dev_df[dev_link_names_xyz].isnull().values.sum()/3)
    dev_df = interpolate_missing_markers(hand_df=dev_df, link_names_separated=dev_link_names_xyz, method='time', verbose=verbose)

    # compute mean location of keyboard
    dev_df['pos_x'] = dev_df[dev_link_names_x].mean(axis=1)
    dev_df['pos_y'] = dev_df[dev_link_names_y].mean(axis=1)
    dev_df['pos_z'] = dev_df[dev_link_names_z].mean(axis=1)
    dev_df['pos'] = dev_df.apply(lambda x: np.array([x.pos_x, x.pos_y, x.pos_z]), axis=1)

    # convert link positions to numerical
    for dev_link_name in dev_link_names:
        df[dev_link_name] = dev_df.apply(lambda x: np.array([x[dev_link_name + '_x'], x[dev_link_name + '_y'], x[dev_link_name + '_z']]), axis=1)
    df['dev_pos'] = dev_df['pos']

    # do not compute arm links positions (thumb3 and thumb4) w.r.t. keyboard position
    # save that for later
    # df['thumb3_dev'] = df.thumb3 - dev_df['pos']
    # df['thumb4_dev'] = df.thumb4 - dev_df['pos']

    return df, stat_dict

# save processed dataframe in h5 format
# ---------------------------------------------------------------------------
# Serialization and diagnostic plots
# ---------------------------------------------------------------------------

def save_processed_h5(data_df, output_dir, data_dir, midi_file=None, restored_sentence=None, verbose=True):
    """Write the merged pandas frame table and optional session metadata."""

    # create target file name as {category_name}_{enrollment_name}_{session_name}.h5
    subject_name = data_dir.split('/')[-5]
    category_name = data_dir.split('/')[-4]
    enrollment_name = data_dir.split('/')[-3]
    session_name = data_dir.split('/')[-2]
    target_file_name = subject_name + '_' + category_name + '_' + enrollment_name + '_' + session_name 

    # save restored sentence as text file
    if restored_sentence is not None:
        with open(os.path.join(output_dir, target_file_name + '_restored_sentence.txt'), 'w') as f:
            f.write(restored_sentence)

    # save midi file
    if midi_file is not None:
        with open(os.path.join(output_dir, target_file_name + '.mid'), 'wb') as outf:
            midi_file.writeFile(outf)

    # save as h5
    data_df.to_hdf(os.path.join(output_dir, target_file_name + '.h5'), key='df', mode='w')
    if verbose:
        print('{} images stored'.format(data_df.shape[0]))

def compute_joint_angle(df, origin, point_a, point_b, joint_name, use_vector=False, invert_joint=False, verbose=True):
    """Derive one joint angle from three tracked points or direction vectors."""
    
    # compute angle between two vectors created by the three points
    df['vector_a'] = df.apply(lambda x: x[point_a] - x[origin], axis=1)
    if use_vector:
        df['vector_b'] = df[point_b].copy()
    else:
        df['vector_b'] = df.apply(lambda x: x[point_b] - x[origin], axis=1)
    #df[joint_name] = df.apply(lambda x: np.arccos(np.dot(x.vector_a,x.vector_b) / (np.linalg.norm(x.vector_a) * np.linalg.norm(x.vector_b))), axis=1)
    df[joint_name] = df.apply(lambda x: np.arctan2(np.linalg.norm(np.cross(x.vector_a,x.vector_b)), np.dot(x.vector_a,x.vector_b)), axis=1)
    
    # invert joint if requested
    if invert_joint:
        df[joint_name] = df[joint_name].apply(lambda x: np.pi - x)

    # bug fix in 25.02 finger_flex03
    # if joint_name == 'jointt6':
    #     df[joint_name] = df[joint_name].apply(lambda x: np.pi - x)

    # find the outliers threshold by calaculating the quantile from 0.0001 to 0.01
    thresh_range = np.concatenate([np.arange(0.0001, 0.001, 0.0001),np.arange(0.001, 0.01, 0.001)])
    thresh_df = pd.DataFrame(data={'threshold':thresh_range})
    thresh_df['quantile'] = thresh_df['threshold'].apply(lambda x: df[joint_name].quantile(x))
    selected_thresh = thresh_df[thresh_df['quantile'] < thresh_df['quantile'].mean()]['quantile'].max()

    # inform outliers if found
    outliers_count = 0
    fixed_thresh = 0.5
    if thresh_df['quantile'].max() - thresh_df['quantile'].min() > 1.0:
        outliers_count = len(df.loc[(df[joint_name] < selected_thresh) | (df[joint_name] < fixed_thresh),joint_name])
        #df.loc[(df[joint_name] < selected_thresh) | (df[joint_name] < fixed_thresh),joint_name] = np.nan
    else:
        outliers_count = len(df.loc[df[joint_name] < fixed_thresh,joint_name])
        #df.loc[df[joint_name] < fixed_thresh,joint_name] = np.nan
    #df = df.dropna()
    if verbose:
        print('Found {} outliers for joint {}'.format(outliers_count, joint_name))

    #plot_joint_angle(df[joint_name].interpolate(method='pad'), joint_name)
    return df.drop(columns=['vector_a','vector_b']), outliers_count


def compute_normal_vector_from_points(df, point_a, point_b, vector_name):
    """Store the normalized direction vector between two tracked points."""

    # compute normal vector for the hand palm
    df[vector_name] = df.apply(lambda x: x[point_b] - x[point_a], axis=1)
    df[vector_name] = df.apply(lambda x: x[vector_name] / np.linalg.norm(x[vector_name]), axis=1)

    return df

def compute_normal_to_plane(df, origin, point_a, point_b, normal_name):
    """Store the unit normal of a plane defined by three tracked points."""

    # compute normal vector for the hand palm
    df['vector_a'] = df.apply(lambda x: x[point_a] - x[origin], axis=1)
    df['vector_b'] = df.apply(lambda x: x[point_b] - x[origin], axis=1)
    df[normal_name] = df.apply(lambda x: np.cross(x.vector_b, x.vector_a) / np.linalg.norm(np.cross(x.vector_a, x.vector_b)), axis=1)

    return df.drop(columns=['vector_a','vector_b'])

def plot_joint_angle(df_col, joint_name):
    """Render one joint-angle trace for preprocessing diagnostics."""

    x = range(df_col.shape[0])
    plt.plot(x, df_col.values, color='b')
    plt.title(joint_name)
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.grid()
    plt.savefig('{}.png'.format(joint_name))

def plot_joint_angles_graphs(data_df, output_dir, data_dir, verbose=True):
    """Save the session-level collection of joint-angle diagnostic plots."""

    if verbose:
        print('Plotting joints angles graphs...')

    # prepare parameters for plotting
    df_columns = data_df.columns.tolist()
    #joint_names = [x for x in df_columns if (x[:6] != 'finger' and x[:5] != 'thumb' and x != 'img' and x[-4:] != 'time') and '_der' not in x]
    joint_names = [x for x in df_columns if (x[:5] == 'joint' or x[:5] == 'wrist') and '_der' not in x]
    n_rows, n_cols = round(len(joint_names) / 4), 4

    # set figure
    x = range(len(data_df))
    fig = plt.figure()
    fig.set_size_inches(5*n_cols,5*n_rows)
    fig.patch.set_facecolor('white')

    # create subplot for each joint
    for i, joint_name in enumerate(joint_names):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(x, data_df[joint_name].values, color='b')
        plt.title(joint_name)
        plt.subplots_adjust(hspace=1.0)
        plt.grid()

    # create target file name as {category_name}_{enrollment_name}_{session_name}_joints.png
    subject_name = data_dir.split('/')[-5]
    category_name = data_dir.split('/')[-4]
    enrollment_name = data_dir.split('/')[-3]
    session_name = data_dir.split('/')[-2]
    target_file_name = subject_name + '_' + category_name + '_' + enrollment_name + '_' + session_name + '_joints.png'

    # save graph plots
    plt.savefig(os.path.join(output_dir, target_file_name), dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process raw data session into H5 file with synchronized multimodal data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to raw data session directory (e.g., data/raw_data/recordings_01/playing/enrollment01/multi_playing01)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Path to output directory for H5 files'
    )

    # Image processing options
    parser.add_argument(
        '--img-res',
        type=int,
        default=224,
        help='Target resolution for ultrasound images (will be resized to img_res x img_res)'
    )
    parser.add_argument(
        '--use-imgs',
        action='store_true',
        help='Include ultrasound images in processing'
    )
    parser.add_argument(
        '--no-resize-imgs',
        action='store_true',
        help='Skip image resizing (keep original size)'
    )

    # Vicon options
    parser.add_argument(
        '--left-handed',
        action='store_true',
        help='Process left hand data instead of right hand'
    )
    parser.add_argument(
        '--no-modified',
        action='store_true',
        help='Use original Vicon CSV instead of modified version'
    )

    # Output options
    parser.add_argument(
        '--no-print-joints',
        action='store_true',
        help='Skip joint angle visualization plots'
    )
    parser.add_argument(
        '--use-calib',
        action='store_true',
        help='Use calibration data for ultrasound time synchronization'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Process arguments
    right_handed = not args.left_handed
    resize_imgs = not args.no_resize_imgs
    modified = not args.no_modified
    print_joints = not args.no_print_joints
    verbose = not args.quiet

    # Print configuration
    if verbose:
        print("="*80)
        print("H5 DATA PROCESSING")
        print("="*80)
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Image resolution: {args.img_res}x{args.img_res}")
        print(f"Include images: {args.use_imgs}")
        print(f"Resize images: {resize_imgs}")
        print(f"Right handed: {right_handed}")
        print(f"Use modified Vicon: {modified}")
        print(f"Print joint plots: {print_joints}")
        print(f"Use calibration: {args.use_calib}")
        print("="*80)
        print()

    # Process the session
    try:
        stat_dict = prepare_data_session(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            img_res=args.img_res,
            use_imgs=args.use_imgs,
            resize_imgs=resize_imgs,
            right_handed=right_handed,
            modified=modified,
            print_joints=print_joints,
            verbose=verbose,
            use_calib=args.use_calib
        )

        if verbose:
            print()
            print("="*80)
            print("PROCESSING COMPLETE")
            print("="*80)
            print("Statistics:")
            for key, value in stat_dict.items():
                print(f"  {key}: {value}")
            print("="*80)

        # Save statistics to CSV
        stats_filename = f"{stat_dict['subject']}_{stat_dict['category']}_{stat_dict['enrollment']}_{stat_dict['session']}_stats.csv"
        stats_path = os.path.join(args.output_dir, stats_filename)

        # Convert dict to DataFrame and save
        stats_df = pd.DataFrame([stat_dict])
        stats_df.to_csv(stats_path, index=False)

        if verbose:
            print(f"\nStatistics saved to: {stats_path}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Processing failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
