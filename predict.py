import os
import sys
import pred

if __name__ == '__main__':
    assert len(sys.argv) == 3
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pred.mai(src_image_dir, save_dir)
    # pred.mai(1, 1)s