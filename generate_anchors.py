import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_anchors_simple(img_rows, img_cols, k):
    aspect_ratios_r = [ 1, 2, 1]
    aspect_ratios_c = [ 2, 1, 1]
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
            anchor = [r, c, int(length*ar), int(length*ac)]
            anchors = np.append(anchors, anchor)

    anchors = np.reshape(anchors, (int(len(anchors)/4), 4))
    return anchors

def main(): 
  A = generate_anchors_simple(224,224,200)
  print(A)
  print(A.shape)

  plt.scatter(A[:,0], A[:,1], )
  plt.show()

if __name__ == "__main__":
    main()