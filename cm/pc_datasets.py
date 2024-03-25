import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import posa.data_utils as du
from tqdm import tqdm





class ProxDataset_txt(Dataset):    # when jump_step=8, for a whole seq, dataset's max_frame is 165, max num_seg is 29
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, max_frame=220,
                 ds_weights_path="posa/support_files/downsampled_weights.npy", jump_step=8, step_multiplier=1, max_objs=8, pnt_size=1024, 
                 objs_data_dir='data/protext/objs', max_cats=13, **kwargs):
        '''
            data_dir: directory that stores processed PROXD dataset.
            fix_orientation: flag that specifies whether we always make the first pose in a motion sequence facing
                             towards a canonical direction.
            no_obj_classes: number of contact object classes.
            max_frame: the maximum motion sequence length which the model accepts (after applying frame skipping).
            ds_weights_path: the saved downsampling matrix for downsampling body vertices.
            jump_step: for every jump_step frames, we only select the first frame for some sequence.
            step_multiplier: a dummy parameter used to control the number of examples seen in each epoch (You can
                             ignore it if you don't know how to adjust it).
        '''
        self.data_dir = data_dir
        self.max_objs = max_objs
        self.pnt_size = pnt_size
        self.max_cats = max_cats
        
        # Setup handle case for dataset: 0 for training, 1 for testing
        is_train = self.data_dir.split('_')[1]
        self.handle = 0 if is_train == 'train' else 1
        self.objs_dir = objs_data_dir
        self.context_dir = os.path.join(data_dir, "context")
        self.reduced_verts_dir = os.path.join(data_dir, "reduced_vertices")
        self.seq_names = [f.split('.txt')[0] for f in os.listdir(self.context_dir)]

        # Setup reading object files and cases
        self._setup_static_objs()

        # Initialize for human sequences
        self.reduced_verts_dict = dict()
        self.context_dict = dict()

        self.total_frames = 0
        for seq_name in self.seq_names:
            self.reduced_verts_dict[seq_name] = torch.tensor(np.load(os.path.join(self.reduced_verts_dir, seq_name + ".npy")), dtype=torch.float32)
            with open(os.path.join(self.context_dir, seq_name + ".txt")) as f:
                text_prompt, given_objs, target_obj = f.readlines()
                text_prompt = text_prompt.strip('\n')
                given_objs = given_objs.strip('\n').split(' ')
                self.context_dict[seq_name] = (text_prompt, given_objs, target_obj)

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)

        self.jump_step = jump_step
        self.step_multiplier = step_multiplier

    @property
    def _cat(self):
        return {
            "chair": 1,
            "table": 2,
            "cabinet": 3,
            "sofa": 4,
            "bed": 5,
            "chest_of_drawers": 6,
            "chest": 6,
            "stool": 7,
            "tv_monitor": 8,
            "tv": 8,
            "lighting": 9,
            "shelving": 10,
            "seating": 11,
            "furniture": 12,
            "human": 0,
        }

    def _setup_static_objs(self):
        self.scenes = os.listdir(self.objs_dir)
        self.objs = dict()
        self.cats = dict()
        for scene in self.scenes:
            self.objs[scene] = dict()
            self.cats[scene] = dict()
            
            objs_list = os.listdir(os.path.join(self.objs_dir, scene))
            for obj_file in objs_list:
                obj = obj_file[:-4]
                cat = obj.split('.')[0].split('_')[0]
                # Read vertices of objects
                with open(os.path.join(self.objs_dir, scene, obj_file), 'rb') as f:
                    verts = np.load(f)
                self.objs[scene][obj] = verts
                self.cats[scene][obj] = self._cat[cat]
            
    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        # seq_idx = torch.randint(len(self.seq_names), size=(1,))
        seq_idx = idx
        seq_name = self.seq_names[seq_idx]
        scene = seq_name.split('_')[0]
        all_objs = self.objs[scene]
        all_cats = self.cats[scene]
        text_prompt, given_objs, target_obj = self.context_dict[seq_name]
        human_verts = self.reduced_verts_dict[seq_name]

        # Initialize for objects, note that, the first object is human
        obj_verts = torch.zeros(self.max_objs+1, self.pnt_size, 3)
        obj_verts[0] = human_verts.clone().detach()
        obj_mask = torch.zeros(self.max_objs+1)
        obj_cats = torch.zeros(self.max_objs+1, self.max_cats)
        obj_cats[0][self._cat['human']] = 1
        for idx, obj in enumerate(given_objs):
            cat = obj.split('_')[0]
            obj_verts[idx+1] = torch.tensor(all_objs[obj])
            obj_mask[idx+1] = 1
            obj_cats[idx+1][self._cat[cat]] = 1

        # Retrieve information of target vertices
        target_verts = all_objs[target_obj]
        target_cat = target_obj.split('_')[0]
        target_num = self._cat[target_cat]
        target_cat = torch.zeros(self.max_cats)
        target_cat[target_num] = 1

        return obj_mask, {"object_verts": obj_verts,"obj_cats": obj_cats, "target_verts": target_verts,"target_cat": target_cat,"text": text_prompt}


class HUMANISE(Dataset):    # when jump_step=8, for a whole seq, dataset's max_frame is 165, max num_seg is 29
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, max_frame=220,
                 ds_weights_path="posa/support_files/downsampled_weights.npy", jump_step=8, step_multiplier=1, max_objs=8, pnt_size=1024, 
                 objs_data_dir='data/humanise/objs', max_cats=11, **kwargs):
        '''
            data_dir: directory that stores processed PROXD dataset.
            fix_orientation: flag that specifies whether we always make the first pose in a motion sequence facing
                             towards a canonical direction.
            no_obj_classes: number of contact object classes.
            max_frame: the maximum motion sequence length which the model accepts (after applying frame skipping).
            ds_weights_path: the saved downsampling matrix for downsampling body vertices.
            jump_step: for every jump_step frames, we only select the first frame for some sequence.
            step_multiplier: a dummy parameter used to control the number of examples seen in each epoch (You can
                             ignore it if you don't know how to adjust it).
        '''
        self.data_dir = data_dir
        self.max_objs = max_objs
        self.pnt_size = pnt_size
        self.max_cats = max_cats
        
        # Setup handle case for dataset: 0 for training, 1 for testing
        is_train = self.data_dir.split('/')[-1]
        self.handle = 0 if is_train == 'train' else 1
        self.objs_dir = objs_data_dir
        self.context_dir = os.path.join(data_dir, "context")
        self.reduced_verts_dir = os.path.join(data_dir, "reduced_vertices")
        self.seq_names = [f.split('.txt')[0] for f in os.listdir(self.context_dir)]

        # Setup reading object files and cases
        self._setup_static_objs()

        # Initialize for human sequences
        self.reduced_verts_dict = dict()
        self.context_dict = dict()

        self.total_frames = 0
        for seq_name in self.seq_names:
            self.reduced_verts_dict[seq_name] = torch.tensor(np.load(os.path.join(self.reduced_verts_dir, seq_name + ".npy")), dtype=torch.float32)
            with open(os.path.join(self.context_dir, seq_name + ".txt")) as f:
                text_prompt, given_objs, target_obj = f.readlines()
                text_prompt = text_prompt.strip('\n')
                given_objs = given_objs.strip('\n').split(' ')
                self.context_dict[seq_name] = (text_prompt, given_objs, target_obj)

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)

        self.jump_step = jump_step
        self.step_multiplier = step_multiplier

    @property
    def _cat(self):
    
        return {
            "bed": 1,		# bed
            "sofa": 2,  		# sofa
            "table": 3,		# table
            "door": 4,  		# door
            "desk": 5,		# desk
            "refrigerator": 6, 		# refrigerator
            "chair": 7,
            "counter": 8,
            "bookshelf": 9,
            "cabinet": 10,
            "human": 0
        }

    def _setup_static_objs(self):
        self.scenes = os.listdir(self.objs_dir)
        self.objs = dict()
        self.cats = dict()
        for scene in self.scenes:
            self.objs[scene] = dict()
            self.cats[scene] = dict()
            
            objs_list = os.listdir(os.path.join(self.objs_dir, scene))
            for obj_file in objs_list:
                obj = obj_file[:-4]
                cat = obj.split('_')[0]
                if cat in self._cat:
                    # Read vertices of objects
                    with open(os.path.join(self.objs_dir, scene, obj_file), 'rb') as f:
                        verts = np.load(f)
                    self.objs[scene][obj] = verts
                    self.cats[scene][obj] = self._cat[cat]
            
    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        
        # seq_idx = torch.randint(len(self.seq_names), size=(1,))
        seq_idx = idx
        seq_name = self.seq_names[seq_idx]
        scene = seq_name[:9] + '_00'
        all_objs = self.objs[scene]
        all_cats = self.cats[scene]
        text_prompt, given_objs, target_obj = self.context_dict[seq_name]
        human_verts = self.reduced_verts_dict[seq_name]

        # Initialize for objects, note that, the first object is human
        obj_verts = torch.zeros(self.max_objs+1, self.pnt_size, 3)
        obj_verts[0] = human_verts.clone().detach()
        obj_mask = torch.zeros(self.max_objs+1)
        obj_cats = torch.zeros(self.max_objs+1, self.max_cats)
        obj_cats[0][self._cat['human']] = 1
        for idx, obj in enumerate(given_objs):
            cat = obj.split('_')[0]
            obj_verts[idx+1] = torch.tensor(all_objs[obj])
            obj_mask[idx+1] = 1
            obj_cats[idx+1][self._cat[cat]] = 1

        # Retrieve information of target vertices
        target_verts = all_objs[target_obj]
        target_cat = target_obj.split('_')[0]
        target_num = self._cat[target_cat]
        target_cat = torch.zeros(self.max_cats)
        target_cat[target_num] = 1

        return obj_mask, {"object_verts": obj_verts,"obj_cats": obj_cats, "target_verts": target_verts,"target_cat": target_cat,"text": text_prompt}
def load_data(*,
    train_data_dir,
    datatype,
    max_frame,
    fix_orientation,
    step_multiplier,
    jump_step,
    batch_size,
    num_workers
    ):
    if datatype == "proxd":
        train_dataset = ProxDataset_txt(train_data_dir, max_frame=max_frame, fix_orientation=fix_orientation,
                                    step_multiplier=step_multiplier, jump_step=jump_step)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_dataset = HUMANISE(train_data_dir, max_frame=max_frame, fix_orientation=fix_orientation,
                                    step_multiplier=step_multiplier, jump_step=jump_step)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)