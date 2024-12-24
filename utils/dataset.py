"""Data fetching
"""
# MIT License
# 
# Copyright (c) 2018 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import random

import torch
from PIL import Image
from multiprocess import Process, Queue

import numpy as np

class DataClass(object):
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.label = label
        self.indices = indices
        return


class Dataset():

    def __init__(self, path=None,mode="target"):
        self.mode = mode
        # 内存吃紧，只存路径
        self.images = []
        self.classes = []

        self.index_queue = None
        self.index_worker = None
        self.batch_queue = None
        self.batch_workers = None
        if path is not None:
            self.init_from_path(path,mode)

    def clear(self):
        del self.images
        del self.classes
        self.release_queue()
        self.__init__()

    def init_from_path(self, path,mode):
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path,mode)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder' % path)
        print('%d images of %d classes loaded' % (len(self.images), len(self.classes)))

    def init_from_folder(self, folder,mode):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        # list的append比array快 先list最后转
        images = []
        total_idx = 0
        for i, class_name in enumerate(class_names):
            label = i
            class_dir = os.path.join(folder, class_name)
            if os.path.isdir(class_dir):
                image_names = os.listdir(class_dir)
                image_paths = [os.path.join(class_dir, img_name) for img_name in image_names]
                # 低于2张图的没法和自己做混淆攻击，忽略
                if mode=="untarget" and len(image_paths) < 2:
                    continue
                # 低于3张图的没法做模拟黑盒，忽略
                if mode == "target" and len(image_paths) < 3:
                    continue
                images.extend(image_paths)
                self.classes.append(DataClass(class_name=class_name,
                                              label=label,
                                              indices=[i for i in range(total_idx, total_idx+len(image_paths))]))
                total_idx += len(image_paths)
        self.images = np.array(images, dtype=np.object_)

    def build_subset_from_classes(self, classes_indices, new_labels=True):
        subset = type(self)()
        subset.mode = self.mode
        images = []
        total_idx = 0
        for i, class_indices in enumerate(classes_indices):
            source_class = self.classes[class_indices]
            images.extend(self.images[source_class.indices])
            subset.classes.append(DataClass(
                class_name=source_class.class_name,
                label=i if new_labels else source_class.label,
                indices=[i for i in range(total_idx, total_idx + len(source_class.indices))]
            ))
            total_idx += len(source_class.indices)
        subset.images = np.array(images, dtype=np.object_)

        print('build subset: %d images of %d classes' % (len(subset.images), len(classes_indices)))
        return subset


    def separate_by_ratio(self, ratio, random_sort=True):
        num_classes_subset1 = int(len(self.classes) * ratio)
        if random_sort:
            classes_indices = np.random.permutation(len(self.classes))
        else:
            classes_indices = np.arange(len(self.classes))
        classes1_indices = classes_indices[:num_classes_subset1]
        classes2_indices = classes_indices[num_classes_subset1:]
        return self.build_subset_from_classes(classes1_indices), self.build_subset_from_classes(classes2_indices)


    # Data Loading
    def init_index_queue(self, batch_format):
        if self.index_queue is None:
            self.index_queue = Queue()
        
        if batch_format in ['random_samples']:
            size = len(self.classes)
            index_queue = np.random.permutation(size)[:,None]
        else:
            raise ValueError('IndexQueue: Unknown batch_format: {}!'.format(batch_format))
        for idx in list(index_queue):
            self.index_queue.put(idx)


    # 从index queue获取分类下标并根据下标提数据
    def get_batch(self, batch_size, batch_format,transforms):
        classes_indices_batch = []
        if batch_format =='random_samples':
            while len(classes_indices_batch) < batch_size:
                classes_indices_batch.extend(self.index_queue.get(block=True, timeout=1000))
            assert len(classes_indices_batch) == batch_size
            sources = []
            targets = []
            sources_name = []
            targets_name = []
            for source_class_indices in classes_indices_batch:
                if self.mode == 'target':
                    # 随机找三张source的
                    # 冒充每个人不能低于三张图就是因为这里
                    assert len(self.classes[source_class_indices].indices) >= 3
                    source_random_indices = random.sample(self.classes[source_class_indices].indices, 3)
                    source = [Image.open(self.images[i]).convert('RGB') for i in source_random_indices]
                    total = len(self.classes)
                    target_class_indices = (source_class_indices+random.randint(1, total-1)) % total
                    # 随机找三张target的
                    assert len(self.classes[target_class_indices].indices) >= 3
                    target_random_indices = random.sample(self.classes[target_class_indices].indices, 3)
                    target = [Image.open(self.images[i]).convert('RGB') for i in target_random_indices]
                elif self.mode == 'untarget':
                    # 混淆每个人不能低于两张图就是因为这里
                    assert len(self.classes[source_class_indices].indices) >= 2
                    source_target_indices = random.sample(self.classes[source_class_indices].indices, 2)
                    source = Image.open(self.images[source_target_indices[0]]).convert('RGB')
                    target_class_indices = source_class_indices
                    target = Image.open(self.images[source_target_indices[1]]).convert('RGB')
                sources.append(source)
                targets.append(target)
                sources_name.append(self.classes[source_class_indices].class_name)
                targets_name.append(self.classes[target_class_indices].class_name)
        else:
            raise ValueError('get_batch: Unknown batch_format: {}!'.format(batch_format))
        if self.mode == 'target':
            source_faces = torch.stack([transforms(row[0]) for row in sources])
            source_faces2 = torch.stack([transforms(row[1]) for row in sources])
            source_faces3 = torch.stack([transforms(row[2]) for row in sources])
            target_faces = torch.stack([transforms(row[0]) for row in targets])
            target_faces2 = torch.stack([transforms(row[1]) for row in targets])
            target_faces3 = torch.stack([transforms(row[2]) for row in targets])
            batch = {
                'sources': source_faces,
                'sources2': source_faces2,
                'sources3': source_faces3,
                'sources_name': sources_name,
                'targets': target_faces,
                'targets2': target_faces2,
                'targets3': target_faces3,
                'targets_name': targets_name,
            }
        elif self.mode == 'untarget':
            source_faces = torch.stack([transforms(img) for img in sources])
            target_faces = torch.stack([transforms(img) for img in targets])
            batch = {
                'sources': source_faces,
                'sources_name': sources_name,
                'targets': target_faces,
                'targets_name': targets_name,
            }
        return batch

    # Multithreading preprocessing images
    def start_index_queue(self, batch_format):
        if batch_format != 'random_samples':
            return
        self.index_queue = Queue()
        def index_queue_worker():
            while True:
                if self.index_queue.empty():
                    self.init_index_queue(batch_format)
                time.sleep(0.5)
        self.index_worker = Process(target=index_queue_worker)
        self.index_worker.daemon = True
        self.index_worker.start()

    def start_batch_queue(self, batch_size, batch_format,transforms, maxsize=50, num_threads=3):
        if self.index_queue is None:
            self.start_index_queue(batch_format)

        self.batch_queue = Queue(maxsize=maxsize)
        def batch_queue_worker(seed):
            np.random.seed(seed)
            while True:
                if self.batch_queue.qsize() < 30:
                    batch = self.get_batch(batch_size, batch_format, transforms)
                    self.batch_queue.put(batch)

        self.batch_workers = []
        for i in range(num_threads):
            worker = Process(target=batch_queue_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)
    
    def pop_batch_queue(self, timeout=50):
        return self.batch_queue.get(block=True, timeout=timeout)
      
    def release_queue(self):
        if self.index_queue is not None:
            self.index_queue.close()
        if self.batch_queue is not None:
            self.batch_queue.close()
        if self.index_worker is not None:
            self.index_worker.terminate()   
            del self.index_worker
            self.index_worker = None
        if self.batch_workers is not None:
            for w in self.batch_workers:
                w.terminate()
                del w
            self.batch_workers = None

