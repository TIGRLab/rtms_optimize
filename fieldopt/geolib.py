#!/usr/bin/env python
# coding: utf-8

# Library of useful geometry functions

# In[ ]:


import numpy as np
from scipy.linalg import lstsq
import gmsh
from simnibs.msh import gmsh_numpy as simgmsh #simnibs gmsh wrapper module


# In[ ]:


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    
    Source: https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array
    """
    if isinstance(vector, np.ndarray):
        return np.array([[0, -vector.item(2), vector.item(1)],
                         [vector.item(2), 0, -vector.item(0)],
                         [-vector.item(1), vector.item(0), 0]])
    else:
        return np.array([[0, -vector[2], vector[1]], 
                         [vector[2], 0, -vector[0]], 
                         [-vector[1], vector[0], 0]])


# In[ ]:


def rotate_vec2vec(v1,v2):
    '''
    Rotate vector v1 --> v2 and return the transformation matrix R that achieves this
    '''
    
    n = np.cross(v1,v2)
    sinv = np.linalg.norm(n)
    cosv = np.dot(v1,v2)
    R = np.eye(3) + skew(n) + np.matmul(skew(n),skew(n))*(1 - cosv)/(sinv**2) 
    return R


# In[ ]:


def quad_fit(X,b):
    '''
    Perform quadratic surface fitting of form:
    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2 
    By finding the least squares solution of Ax = b
    '''
    
    #Formulate the linear problem
    A = np.c_[
        
        np.ones((X.shape[0],1)),
        X[:,:2],
        np.prod(X[:,:2],axis=1),
        X[:,:2]**2        
    ]
    
    C,_,_,_ = lstsq(A,b)
    return C


# In[ ]:


def compute_principal_dir(x,y,C):
    '''
    Compute the principal direction of a quadratic surface of form:
    S: f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
    Using the second fundamental form basis matrix eigendecomposition method
    
    x -- scalar x input
    y -- scalar y input
    C -- scalar quadratic vector (a,b,c,d,e,f)
    
    V[:,0] -- major principal direction
    V[:,1] -- minor principal direction
    n      -- normal to surface S at point (x,y)
    '''
    
    #Compute partial first and second derivatives
    r_x = np.array([1, 0, 2*C[4]*x + C[1] + C[3]*y])
    r_y = np.array([0, 1, 2*C[5]*y + C[2] + C[3]*x])
    r_xx = np.array([0, 0, 2*C[4]])
    r_yy = np.array([0, 0, 2*C[5]])
    r_xy = np.array([0, 0, C[3]])
    
    #Compute surface point normal
    r_x_cross_y = np.cross(r_x,r_y)
    n = r_x_cross_y/np.linalg.norm(r_x_cross_y)
    
    #Compute second fundamental form constants
    L = np.dot(r_xx,n)
    M = np.dot(r_xy,n)
    N = np.dot(r_yy,n)
    
    #Form basis matrix
    P = np.array([
        [L, M],
        [M, N]
    ])
    
    #Eigendecomposition, then convert into 3D vector
    _,V = np.linalg.eig(P)
    V = np.concatenate((V,np.zeros((1,2))), axis=0)
    
    return V[:,0], V[:,1], n


# In[ ]:


def interpolate_angle(u,v,t,l=90.0):
    '''
    Perform cosine angle interpolation between two orthogonal vectors u and v
    
    u -- 3D vector
    v -- 3D vector orthogonal to u
    t -- interpolation value
    l -- period of rotation
    '''
    
    theta = (t/l)*(np.pi/2)
    p = np.r_[u*np.cos(theta) + v*np.sin(theta), 0]
    return p


# In[ ]:


def quadratic_surf(x,y,C):
    '''
    Compute a quadratic function of form:
                    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
                    
    x -- scalar x input
    y -- scalar y input
    C -- quadratic constants vector (a,b,c,d,e,f)
    '''
    
    return C[0] + C[1]*x + C[2]*y + C[3]*x*y + C[4]*x*x + C[5]*y*y


# In[ ]:


def map_param_2_surf(x,y,C):
    '''
    For some mesh-based surface S, define a parameterization using a quadratic fit
                    f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
    
    Compute the mapping from (x,y) --> (x,y,f(x,y))
    
    x -- scalar x input
    y -- scalar y input
    C -- scalar quadratic constants vector (a,b,c,d,e,f)
    '''
    
    #Compute approximate surface at (x,y)
    z = quadratic_surf(x,y,C)
    
    #Form input vector
    v = np.array([x,y,z],dtype=np.float64)
    
    return v


# In[ ]:


def map_rot_2_surf(x,y,t,C):
    '''
    For some mesh-based surface S, define a least squares quadratic surface parameterization S':
                        f(x,y) = a + bx + cy + dxy + ex^2 + fy^2
    
    Compute a mapping from (x,y,z,t) --> p which is an interpolated direction vector between the
    two principal directions on S' using t:[0,180]. Both principal directions are defined on the 
    (x,y) plane so we need to align the orientation vector p to the surface normal. This is done 
    using the following steps:
    1. Find rotation R that maps the standard basis z axis to the normal of surface S' at point (x,y)
    2. Apply the rotation to vector p to align it to the new basis defined by the normal yielding p'  
    
    x -- scalar x input
    y -- scalar y input
    t -- interpolation angle [0,180] between 2 principal directions
    C -- scalar quadratic constants vector (a,b,c,d,e,f)

    '''
    
    v1, v2, n = compute_principal_dir(x,y,C)
    p = interpolate_angle(v1[:2],v2[:2],t)
    
    z = np.array([0,0,1],dtype=np.float64)
    R = rotate_vec2vec(z,n)
    pp = np.matmul(R,p)
    
    
    return pp, n


# In[ ]:


def load_gmsh_nodes(gmshpath):
    '''
    Given a fullpath to some .msh file, load in the mesh nodes IDs, triangles and coordinates.
    
    gmshpath -- path to gmsh file
    dimtag   -- tuple specifying the (dimensionality,tagID) being loaded
    
    
    If entity=(dim,tag) not provided then pull the first entity and return
    '''
    
    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, coords, params = gmsh.model.mesh.getNodes(entity[0],entity[1])
    coords = np.array(coords).reshape((len(coords)//3,3))
    
    return nodes, coords, params


# In[ ]:


def load_gmsh_elems(gmshpath, entity):
    
    gmsh.initialize()
    gmsh.open(gmshpath)
    nodes, elem_ids, node_maps = gmsh.model.mesh.getElements(entity[0],entity[1])
    
    return nodes, elem_ids[0], node_maps


# In[ ]:


def define_coil_orientation(loc,rot,n):
    '''
    Construct the coil orientation matrix to be used by simnibs
    loc -- center of coil
    rot -- vector pointing in handle direction (tangent to surface)
    n   -- normal vector
    '''
    
    y = rot/np.linalg.norm(rot)
    z = n/np.linalg.norm(n)
    x = np.cross(y,z)
    c = loc
    
    matsimnibs = np.zeros((4,4), dtype=np.float64)
    matsimnibs[:3, 0] = x
    matsimnibs[:3, 1] = y
    matsimnibs[:3, 2] = -z
    matsimnibs[:3, 3] = c
    matsimnibs[3, 3] = 1
    
    return matsimnibs


# In[ ]:


def get_field_subset(field_msh, tag_list):
    '''
    From a .msh file outputted from running a TMS field simulation extract the field magnitude
    values of elements provided for in tag_list
    
    field_msh  --  Path to .msh file result from TMS simulation
    tag_list   --  List of element tags to use as subset 
    
    Output:
    normE      --  List of electric field norms (magnitudes) subsetted according to tag_list
    '''
    
    msh = simgmsh.read_msh(field_msh)
    norm_E = msh.elmdata[1].value
    return norm_E[tag_list]


# In[ ]:


def compute_field_score(normE, proj_map, parcel):
    '''
    From a list of field magnitudes in <normE> compute the weighted sum determined
    by the partial parcel weightings in proj_map
    
    Arguments:
    normE      --  1D array of relevant field magnitudes in the order of proj_map 
    proj_map   --  Array of size (len(<normE>), max(parcel ID)) where each row corresponds to an 
                   element (pairing with <normE>) and each column corresponds to a parcel ID. 
    parcel     --  The parcel to compute a score over. Range = [0,max(parcel_ID)]
    
    Output:
    score      --  A single scalar value representing the total stimulation 
    '''
    
    parcel_map = proj_map[:,parcel]
    return np.dot(parcel_map,normE)

