from helper.config import box_constant

#FACE VERIFIACTION HELPER FUNCTIONS

#Validates if the current detected box is valid or not
def verify_box(box, vbox):
        flag = 0

        ## VALIDATE NEW BOX ##
        if box[0] < vbox[0]:
            flag+=1
        if box[1] < vbox[1]:
            flag+=1
        if box[2] > vbox[2]:
            flag+=1
        if box[3] > vbox[3]:
            flag+=1

        ## IF ANY ARE OUTSIDE THE BOX THEN IT IS INVALID ##
        # print(flag)
        if flag >= 1:
            return False
        else:
            return True

#Create a boundary for valid box
def resize_box(w, h, box):

    w = w
    h = h

    ## GET DIMENSIONS ##
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = box
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    #Get new dimensions
    new_width = width * box_constant
    new_height = height * box_constant

    margin_width = (new_width - width) / 2
    margin_height = (new_height - height) / 2

    ## COMPARE COORDINATES

    #Compare topright y axis
    new_top_left_x = top_left_x - margin_width
    if new_top_left_x >= 0:
        top_left_x = int(new_top_left_x)

    if new_top_left_x < 0:
        top_left_x = 0

    #Compare topright x axis
    new_top_left_y = top_left_y - margin_height
    if new_top_left_y >= 0:
        top_left_y = int(new_top_left_y)
    
    if new_top_left_x < 0:
        top_left_y = 0

    #Compare bottomleft y axis
    new_bottom_right_x = bottom_right_x + margin_width
    if new_bottom_right_x <= w:
        bottom_right_x = int(new_bottom_right_x)

    if new_bottom_right_x > w:
        bottom_right_x = w

    #Compare bottomleft x axis
    new_bottom_right_y = bottom_right_y + margin_height
    if new_bottom_right_y <= h:
        bottom_right_y = int(new_bottom_right_y)

    if new_bottom_right_y > h:
        bottom_right_y = h

    resized_box = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    return resized_box

