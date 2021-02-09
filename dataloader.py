import random
import numpy as np
class RECOMBdata(data.Dataset):
    def __init__(self,images,perm,var_sizes,step_size):
        self.images=images
        self.perm=perm
        self.var_sizes=var_sizes
        self.step_size=step_size
    def __len__(self):
        return self.step_size
    def __getitem__(self,index):
        op_type,y0,y1,y_ground_truth,image_index=generate_comb_data(self.var_sizes,self.perm)
        return op_type,y0,y1,y_ground_truth,self.images[image_index]
class SCANdata(data.Dataset):
        def __init__(self,one_hots,images):
            self.one_hots=one_hots
            self.images=images
        def __len__(self):
            return len(self.images)
        def __getitem__(self,index):
            one_hot=self.one_hots[index]
            image=self.images[index]        
            return one_hot,image
class DAEdata(data.Dataset):
        def __init__(self,hsv_images):
            self.hsv_images=hsv_images
            self.image_size=hsv_images[0].shape
        def __len__(self):
            return len(self.hsv_images)
        def __getitem__(self,index):
            ground_truth=self.hsv_images[index]        
            h0,h1=sorted(random.sample(range(0,80),2))
            w0,w1=sorted(random.sample(range(0,80),2))
            blocked_image=np.copy(ground_truth)
            blocked_image[:,h0:h1,w0:w1]=0.0
            return ground_truth,blocked_image
class VAEdata(data.Dataset):
        def __init__(self,hsv_images):
            self.hsv_images=hsv_images
            self.image_size=hsv_images[0].shape
        def __len__(self):
            return len(self.hsv_images)
        def __getitem__(self,index):
            ground_truth=self.hsv_images[index]        
            return ground_truth
def get_image( obj_color, wall_color, floor_color, obj_id,perm):
    index = obj_color + wall_color * 16 + floor_color * 16 * 16 + obj_id * 16 * 16 * 16
    return perm[index]

def get_image_random(obj_color,wall_color,floor_color,obj_id,perm):
    if(obj_color==-1):
        obj_color=random.randint(0,15)
    if(wall_color==-1):
        wall_color=random.randint(0,15)
    if(floor_color==-1):
        floor_color=random.randint(0,15)
    if(obj_id==-1):
        obj_id=random.randint(0,2)
    index = obj_color + wall_color * 16 + floor_color * 16 * 16 + obj_id * 16 * 16 * 16
    return perm[index]

def not_same_random(low,upper,same):
    random_num=random.randint(low,upper)
    while(random_num==same):
        random_num=random.randint(low,upper)
    return random_num

def not_same_random_list(low,upper,list1):
    random_num=random.randint(low,upper)
    while(random_num in list1):
        random_num=random.randint(low,upper)
    return random_num

def one_hot_to_label(one_hot):
    indexes=np.where(one_hot>0)[0]
    obj_color=-1
    wall_color=-1
    floor_color=-1
    obj=-1
    for elem in indexes:
        if (0<elem<16):
            obj_color=elem
        elif (15<elem<32):
            wall_color=elem%16
        elif (31<elem<48):
            floor_color=elem%32
        elif (47<elem<51):
            obj=elem%48
    #obj_color=indexes[0]
    #wall_color=indexes[1]%16
    #floor_color=indexes[2]%32
    #obj=indexes[3]%48
    return [obj_color,wall_color,floor_color,obj]

def generate_comb_data(var_sizes,perm1):
    ## -1 means undefined; according to combination rules we fill it
    y0=[-1,-1,-1,-1]
    y1=[-1,-1,-1,-1]  
    y_ground_truth=[-1,-1,-1,-1]
    y_gt_im=[-1,-1,-1,-1]
    perm=np.random.permutation(4)  #randomly permute the selected qualities to ensure equal representation
    op_type=random.randint(0,2)   #randomly pick between and,in_common,difference
    op_type_ar=np.zeros(3,dtype=np.float)  #turn to one hot
    op_type_ar[op_type]=1.0  # set desired operation
    if not (op_type):
        y0_ngram=random.randint(1,3)   # if op_type == 0 can only be 1 to 3 ngram
        y1_ngram=random.randint(1,4-y0_ngram) #we pick so the and can at most create 4 grams without conflict
        y0_val_picks=perm[:y0_ngram] # we select which properties will be defined for y0
        y1_val_picks=perm[y0_ngram:y0_ngram+y1_ngram] # we select which properties will be defined for y1 which is not in common with y0
        for item in y0_val_picks:   
            y0[item]=random.randint(0,var_sizes[item]) # we set the selected quality to the allowed variations
        for item in y1_val_picks:
            y1[item]=random.randint(0,var_sizes[item])
        for i in y0_val_picks:
            y_ground_truth[i]=y0[i] #ground truth is set so y0 qualities pass down
        for i in y1_val_picks:
            y_ground_truth[i]=y1[i]
            y_gt_im[i]=y1[i] #ground truth is set so 1 qualities pass down
        for item in range(len(y_ground_truth)):
            if (y_ground_truth[item]==-1): # if the ground truth is left undefined we pick a random value to generate a random latent space
                y_gt_im[item]=random.randint(0,var_sizes[item]) 
        image_index=get_image(y_gt_im[0],y_gt_im[1],y_gt_im[2],y_gt_im[3],perm1) # return a image to generate BVAE latent space
        #print("optype0 " + ,y_gt_im[3])
    elif op_type==1: # incommon operator data
        perm=np.random.permutation(4) #randomly shuffle qualities
        common_count=random.randint(1,3) #pick how many in common there will be
        y0_ngram=random.randint(common_count,3) #pick how many qualities will be defined for first vector
        y1_ngram=random.randint(common_count,3) # pick how many qualities will be defined for the second vector
        commons=perm[:common_count] # pick the common qualities
        non_commons=perm[common_count:] # pick the non common qualities
        for pick in commons:
            pick_value=random.randint(0,var_sizes[pick]) # pick the value for the quality
            y0[pick]=pick_value #set value for both vectors
            y1[pick]=pick_value
        for val in non_commons:
            y0_include=random.randint(0,1) #decide if the uncommon quality will exist in first vector
            y1_include=random.randint(0,1) #decide if the uncommon quality will exist in second vector
            if (y0_include):
                y0[val]=random.randint(0,var_sizes[val]) # if it is in the first vector set the value for the quality
                if (y1_include):
                    y1[val]=not_same_random(0,var_sizes[val],y0[val]) # if it is included in the second vector pick a different quality from the first
            elif (y1_include):
                y1[val]=random.randint(0,var_sizes[val]) #if it is not included in y0 pick a random value
        for item in commons:
            y_ground_truth[item]=y0[item] # ground truth includes in common values
            y_gt_im[item]=y0[item]
        for val in non_commons:
            y_gt_im[val]=not_same_random_list(0,var_sizes[val],[y0[val],y1[val]]) #if it is non_common the ground truth value will be different than both
        image_index=get_image(y_gt_im[0],y_gt_im[1],y_gt_im[2],y_gt_im[3],perm1) #return an image to generate BVAE latent space
        
    else: #different value in commons
        y0_picks=perm[:3] # 3 gram vector is selected
        y1_ind=random.randint(0,2) # 
        y1_picks=[perm[y1_ind]]
        for index in y0_picks:
            y0[index]=random.randint(0,var_sizes[index])
            y_ground_truth[index]=y0[index]
        y1[y1_picks[0]]=y0[y1_picks[0]]
        y_ground_truth[y1_picks[0]]=-1
        y_gt_im[y1_picks[0]]=not_same_random(0,var_sizes[y1_picks[0]],y1[y1_picks[0]])
        y_gt_im[perm[3]]=random.randint(0,var_sizes[perm[3]])
        image_index=get_image(y_gt_im[0],y_gt_im[1],y_gt_im[2],y_gt_im[3],perm1)
        #print("optype2 " + ,y_gt_im[3])
    #print(op_type)
    return op_type_ar,one_hot_generate(y0),one_hot_generate(y1),one_hot_generate(y_ground_truth),image_index
def generate_one_hots(dataset_size):
    one_hots=[]
    for index in range(dataset_size):
        one_hots.append(index_to_one_hot(index))
    one_hots=np.array(one_hots)
    return one_hots
            