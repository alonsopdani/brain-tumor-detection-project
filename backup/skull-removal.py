'''
    # convert all images from gray to color
    def gray2bgr(list_of_images):
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in list_of_images]

    images_y, images_n = img_to_gray_scale(images_y), img_to_gray_scale(images_n)

    # skull removal
    def skull_removal(pic):
        gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        ret, markers = cv2.connectedComponents(thresh)
        #Get the area taken by each component. Ignore label 0 since this is the background.
        marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
        
        #Get label of largest component by area
        largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        

        #Get pixels which correspond to the brain
        brain_mask = markers==largest_component
        brain_out = pic.copy()
        #In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[brain_mask==False] = (0,0,0)
        brain_mask = np.uint8(brain_mask)
        kernel = np.ones((8,8),np.uint8)
        closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        brain_out = pic.copy()
        #In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[closing==False] = (0,0,0)
        return brain_out

    # function to apply the previous one in the whole list
    def skull_removal_list(list_of_images):
        return [skull_removal(img) for img in list_of_images]

    images_y, images_n = skull_removal_list(images_y), skull_removal_list(images_n)
    '''