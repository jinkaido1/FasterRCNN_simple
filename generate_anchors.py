import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_anchors_simple(img_rows, img_cols, k):
    aspect_ratios_r = [ 1, 2, 1]
    aspect_ratios_c = [ 2, 1, 1]
    #aspect_ratios_r = [ 1 ]
    #aspect_ratios_c = [ 1 ]
    num_boxes = len( aspect_ratios_c)
    sqrt_k = np.sqrt(k)
    length = np.min([(int)(img_rows/sqrt_k), (int)(img_cols/sqrt_k)])
    row_sampling = img_rows/(np.sqrt(k*1.0/num_boxes))
    col_sampling = img_cols/(np.sqrt(k*1.0/num_boxes))
    rows = np.arange(row_sampling/2, img_rows, row_sampling)
    cols = np.arange(col_sampling/2, img_cols, col_sampling)
    row_t = np.tile( rows.astype(int), len(cols))
    col_t = np.repeat( cols.astype(int), len(rows))
    R = np.transpose([row_t, col_t])
    anchors = np.empty((0,4))
    
    for r,c in R:
        for [ar, ac] in zip(aspect_ratios_r, aspect_ratios_c):
            len_c = int(length*ac)
            len_r = int(length*ar)
            c_fixed = int(max(c-len_c/2, 0))
            r_fixed = int(max(r-len_r/2, 0))
            #anchor = [c, r, len_c, len_r]
            #print(anchor)
            anchor = [c_fixed, r_fixed,\
                      min(len_c, img_cols-c_fixed),\
                      min(len_r, img_rows-r_fixed)]
            #print(anchor)
            #input('s')
            anchors = np.append(anchors, anchor)

    anchors = np.reshape(anchors, (int(len(anchors)/4), 4))
    anchors = anchors.astype(int)
    return anchors

def main(): 
  A = generate_anchors_simple(224,224,200)
  print(A)
  print(A.shape)

  plt.scatter(A[:,0], A[:,1], )
  plt.show()

if __name__ == "__main__":
    main()