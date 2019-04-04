#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numba
import numpy as np


# In[3]:


#Why does parallel break?
@numba.njit
def map_nodes(x, prop_array,out):
    '''
    Convenience function to remap a value according to a properties array
    Arguments:
        x                           Array
        prop_array                  A properties array in the desired order
        
    Description of prop_array
    
    prop_array is an (nx3) numpy array that stores the following information for the ith gmsh element
    [1] - minimum element node number
    [2] - maximum element node number
    [3] - number of element nodes
    
    This will remap the values such that it goes from  0 --> np.sum(prop_array[:,2]) 
    So that array-index based hashing can be used for fast coordinate mapping
    '''
    
    
    #Loop
    for i in np.arange(x.shape[0]):
        for j in np.arange(0,x.shape[1]):
            for k in np.arange(0,prop_array.shape[0]):
        
                if (x[i,j] >= prop_array[k,0]) & (x[i,j] <= prop_array[k,1]):
                    out[i,j] = x[i,j] - prop_array[k,0] + np.sum(prop_array[:k,2])

    return out


# In[4]:


@numba.njit
def homogenous_transform(coords,L):
    '''
    Transform into homogenous coordinates and apply linear map, will modify input!
        coords                              (1x3) array to transform
        L                                   Linear map to apply
    '''
    #Need to account for weird array behaviour
    
    tmp = np.ones((4,coords.shape[0]),dtype=np.float64)
    tmp[:3,:] = coords.T
    #tmp = L @ tmp
    tmp = np.dot(L,tmp)
    coords = tmp[:3,:].T
    
    return coords


# In[5]:


@numba.njit
def meshgrid(x,y,z):
    '''
    Create a mesh-grid using values in x,y,z - all arrays must be of same length
        x                                   X-coordinate array
        y                                   Y-coordinate array
        z                                   Z-coordinate array
    Returns a [3,n] matrix of all points within cubic grid
    '''
    #Create output array of all possible combinations
    mg = np.zeros((3,x.size*y.size*z.size),np.int32)
    
    #For each item in x
    counter = 0 
    for i in np.arange(0,x.size):
        for j in np.arange(0,y.size):
            for k in np.arange(0,z.size):
                
                mg[0,counter] = x[i]
                mg[1,counter] = y[j]
                mg[2,counter] = z[k]
                counter+=1
    return mg


# In[6]:


@numba.njit
def aabb_voxels(coords):
    '''
    Use axis-aligned boundary box in voxel space to identify candidate voxels
        coords                              (4,3) array containing tetrahedral coordinates in voxel space
    '''
    
    #Pre-allocate and store bounds
    min_vox = np.zeros((3),np.int32)
    max_vox = np.zeros((3),np.int32)
    
    #Get min, max then floor and ceil respectively
    for i in np.arange(0,3):
        min_vox[i] = np.min(coords[:,i])
        max_vox[i] = np.max(coords[:,i])
    min_vox = np.floor(min_vox)
    max_vox = np.floor(max_vox)

    #Get voxel set
    x_range = np.arange(min_vox[0],max_vox[0]+1,1,np.int32)
    y_range = np.arange(min_vox[1],max_vox[1]+1,1,np.int32)
    z_range = np.arange(min_vox[2],max_vox[2]+1,1,np.int32)
    vox_arr = meshgrid(x_range,y_range,z_range)

    return vox_arr
    


# In[7]:


@numba.njit
def uniform_tet(coords):
    '''
    Argument:
        coords                A (4,3) matrix with rows representing nodes
    Output:
        point                 A random point inside the tetrahedral volume
    '''
    
    s = np.random.uniform(0,1)
    t = np.random.uniform(0,1)
    u = np.random.uniform(0,1)

    #First cut
    if (s+t > 1):
        s = 1.0 - s
        t = 1.0 - t
        
    #Second set of cuts  
    if (t+u > 1):
        tmp = u
        u = 1.0 - s - t
        t = 1.0 - tmp
    elif (s + t + u > 1):
        tmp = u 
        u = s + t + u - 1 
        s = 1 - t - tmp
        
    a = 1 - s - t - u

    return a*coords[0] + s*coords[1] + t*coords[2] + u*coords[3]


# In[8]:


@numba.njit
def point_in_vox(point,midpoint,voxdim=1):
    '''
    Arguments:
        point                         Iterable of length 3
        midpoint                      Voxel midpoint
        voxdim                        Voxel dimensions, assuming isotropic
        
    Output:
        Boolean: True if point in voxel bounds
    '''
    
    #Shift midpoint upwards by half a voxel (left,top,back --> centre of cube)
    halfvox = voxdim/2.
    midpoint = midpoint + halfvox
    
    #Checks
    if (point[0] <= midpoint[0] - halfvox) or (point[0] >= midpoint[0] + halfvox):
        return False
    elif (point[1] <= midpoint[1] - halfvox) or (point[1] >= midpoint[1] + halfvox):
        return False
    elif (point[2] <= midpoint[2] - halfvox) or (point[2] >= midpoint[2] + halfvox):
        return False
    else:
        return True


# In[9]:


@numba.njit
def estimate_partial_parcel(coord,vox,parcels,out,n_iter=300):
    '''
    Arguments:
        coord               (4,3) indexable iterable of tetrahedral coordinates
        vox                 (n,3) indexable iterable of voxel coordinates
        parcels             (n,1) indexable iterable of parcel labels associated with jth voxel coordinate
        out                 A reference to an array (slice) to be written into
        iter                 Number of Monte-Carlo sampling interations

    For each tetrahedron we want to assign the value of the voxel 
    '''
    
    #Check degenerate case
    if np.unique(parcels).shape[0] == 1:
        out[parcels[0]] = 1
    
    #Shift tetrahedron to origin
    t = coord[0]
    coord = coord - t
    
    #Perform fixed monte carlo sampling
    for i in np.arange(0,n_iter):
        p = uniform_tet(coord)
        for j in np.arange(0,vox.shape[1]):
            
            #If point is in voxel, then move on
            if point_in_vox(p+t, vox[:,j]):
                out[parcels[j]] += 1
                break


# In[10]:


@numba.njit(parallel=True)
def tetrahedral_projection(node_list,coord_arr,ribbon,affine,n_iter=300):
    '''
    Perform tetrahedral projection
        node_list                           List of tetrahedral nodes
        coord_arr                           Coordinate list (length=n) in groups of 3 for each node
        ribbon                              3D array containing parcels
        affine                              Affine transformation matrix associated with ribbon
    '''
    
    #Compute inverse affine
    inv_affine = np.linalg.inv(affine)

    #Loop tetrahedrons
    num_elem=node_list.shape[0]

    #Total number of parcels
    num_parc = int(ribbon.max())
    
    #make output array
    out_arr = np.zeros((num_elem,num_parc+1), dtype=np.float64)
    
    for i in numba.prange(0,num_elem):

        #Get coordinates for nodes
        t_coord = np.zeros((4,3),dtype=np.float32)
        t_coord[0,:] = coord_arr[3*node_list[i,0]:(3*node_list[i,0])+3]
        t_coord[1,:] = coord_arr[3*node_list[i,1]:(3*node_list[i,1])+3]
        t_coord[2,:] = coord_arr[3*node_list[i,2]:(3*node_list[i,2])+3]
        t_coord[3,:] = coord_arr[3*node_list[i,3]:(3*node_list[i,3])+3]

        #Step 1: Transform coordinates to MR space
        t_coord[0:1,:] = homogenous_transform(t_coord[0:1,:],inv_affine)
        t_coord[1:2,:] = homogenous_transform(t_coord[1:2,:],inv_affine)
        t_coord[2:3,:] = homogenous_transform(t_coord[2:3,:],inv_affine)
        t_coord[3:4,:] = homogenous_transform(t_coord[3:4,:],inv_affine)

        #Step 2: Perform axis-aligned boundary box finding
        vox_arr = aabb_voxels(t_coord)
        
        #Step 3: Get parcel values associated with voxels
        parcels = np.zeros((vox_arr.shape[1] + 1),np.int32)
        for j in np.arange(vox_arr.shape[1]):
            parcels[j] = ribbon[vox_arr[0,j],vox_arr[1,j],vox_arr[2,j]]
            
        #Step 4: Estimate partial parcels
        estimate_partial_parcel(t_coord,vox_arr,parcels,out_arr[i,:],n_iter)        

    return out_arr/n_iter    

