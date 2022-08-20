import numpy as np
import cv2
import matplotlib.pyplot as plt


def replaceZeroes(data):
    min_nonzero =  min ( data [np.nonzero(data)] )
    data [ data ==  0 ]  = min_nonzero
    return data

def SSR(src_img , size):
    L_blur = cv2.GaussianBlur(src_img,(size, size),0)
    img = replaceZeroes ( src_img ) 
    L_blur = replaceZeroes(L_blur)
    dst_Img = cv2.log(img /255.0)
    dst_Lblur = cv2.log(L_blur /255.0 )
    dst_IxL = cv2.multiply(dst_Img , dst_Lblur )
    log_R = cv2.subtract(dst_Img, dst_IxL )
    dst_R = cv2.normalize(log_R ,None ,0 ,255 ,cv2.NORM_MINMAX )
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_R,dst_IxL

def plotim(im2):
    im2 = cv2.cvtColor(np.uint8(im2), cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im2)
def replaceZeroes ( data ):
    min_nonzero =  min( data [ np . nonzero ( data ) ] )
    data [data == 0] = min_nonzero
    return data


def MSR( img , scales ):
    weight =  1  / 3.0
    scales_size = len(scales)
    h , w = img.shape[ : 2 ]
    log_R = np.zeros( ( h , w ) , dtype=np.float32 )

    for i in  range ( scales_size ) :
        img = replaceZeroes ( img )
        L_blur = cv2.GaussianBlur( img ,  ( scales [ i ] , scales [ i ] ) ,  0 )
        L_blur = replaceZeroes ( L_blur )
        dst_Img = cv2.log( img /255.0)
        dst_Lblur = cv2.log( L_blur /255.0 )
        dst_Ixl = cv2.multiply ( dst_Img , dst_Lblur )
        log_R+= weight * cv2.subtract( dst_Img , dst_Ixl )

    dst_R = cv2.normalize( log_R , None ,  0 ,  255 , cv2 . NORM_MINMAX )
    log_uint8 = cv2 . convertScaleAbs ( dst_R )
    return log_uint8


if __name__ == '__main__':
    img =  r"C:\Users\rom21\OneDrive\Desktop\git_project\dogs.jpeg"
    size =  3 
    src_img = cv2.imread(img)
    scales =  [ 15 , 101 , 301 ]   # [3,5,9] #Can't see the difference in the effect

    b_gray, g_gray ,r_gray = cv2.split(src_img)
    b_gray, lb = SSR(b_gray , size )
    g_gray, lg = SSR(g_gray, size )
    r_gray, lr = SSR(r_gray, size )
    l = cv2.merge([lb, lg, lr])
    result = cv2.merge([b_gray, g_gray, r_gray])

    r = cv2.multiply(result, l )
    r = cv2.normalize(r ,None ,0 ,255 ,cv2.NORM_MINMAX )
    log_uint8 = cv2.convertScaleAbs(r)
    plotim(log_uint8)
    plotim(result)
    plotim(l)
    pass
    cv2.imshow('img' , src_img)
    cv2.imshow('result', result*0.613)
    cv2.waitKey( 0 )
    cv2.destroyWindow('P')