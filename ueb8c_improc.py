# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:27:57 2018

@author: slavie
"""

import pgm
import numpy as np
import traceback

class IMProcess():
    def __init__(self, width, height, length):
        self.__init__(width, height, length)
    
    def init(self, width, height, length):
        imgshape = (width, height)
        
        self.sum_x = np.zeros((width), dtype=np.int32)
        
    def process_pgm(filename):
        import matplotlib.pyplot as plt

class VideoFile():
    def __init__(self, filename):
        try:
            self.video = pgm.PGMReader(filename)
        except:
            traceback.print_exc()
        else:
            self.filename = filename
            self.init_video(self)
    
    def init_video(self):
        self.frame_no = 0
        self.video.seek_frame(self.frame_no)
        self.length = self.video.length
        self.width = self.video.width
        self.height = self.video.height
        self.img_buffer = self.video.img_buffer
        
    def process_frame(self, frame_no):
        self.frame_no = frame_no
        self.video.seek_frame(self, frame_no)


def process_pgm (filename, n):
    import matplotlib.pyplot as plt
    video = pgm.PGMReader(filename)
    fig = plt.figure(1)
    plt.clf()
    nth = np.int(np.floor(video.length/n))
    for i, frame_no in enumerate(range(0, video.length, nth),1):
        video.seek_frame(frame_no)
        ax = fig.add_subplot(n, 1, i)
        ix = video.img_buffer<50
        video.img_buffer[ix]=255
        if i==1:
            plt.title('Frame=%d W=%d H=%d'%(video.length, video.width, video.height))
        plt.title('Frame'+str(i))
    ax.set_axis_off()    
    plt.show()
    return i

import unittest

class Tests(unittest.TestCase):
    def test_1(self):
        video = pgm.PGMReader('sakkade.pgm')
        self.assertEqual(3, process_pgm('sakkade.pgm', 3))
    def test_2(self):
        self.assertEqual(5, process_pgm('sakkade.pgm', 5))
        
if __name__ == '__main__':
    unittest.main()