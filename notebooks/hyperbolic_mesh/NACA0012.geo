////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CREATE 2DNACA AIRFOIL
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// GENERATE POINTS
c = 1.008930411365;
t = 0.12;  // NACA0012 specification
npoints_surface = 250;
dx = c/(npoints_surface-1);
For i In {0:npoints_surface-2}
	x[i] = c*(i/(npoints_surface-1))^2;
	z[i] = c*5*t*(0.2969*Sqrt(x[i]) - 0.1260*x[i] - 0.3516*x[i]^2.0 + 0.2843*x[i]^3.0 - 0.1015*x[i]^4.0);
EndFor

// GENERATE UPPER CURVE OF NACA SURFACE
p = newp;
j = 0;
For i In {0: #x[]-1}
	Point(i) = {x[i]-c/2, 0.0, z[i]};
	j = j+1;
EndFor

p = newp;
// GENERATE LOWER CURVE OF NACA SURFACE
For i In {1: #x[]-2}
	j = (npoints_surface-1)-i;
	Point(p+i-1) = {x[j]-c/2, 0.0, -z[j]};
EndFor

// CREATE ARRAY OF POINTS FOR B-SPLINE OF THE NACA AIRFOIL
npoints_airfoil = 2*(npoints_surface-2)-1;
spline_idxs = {npoints_surface-1:npoints_airfoil, 0, 1:npoints_surface-2};
Spline(1) = {spline_idxs[]};

// COMPUTE TANGENT CIRCLE TO GENERATE SMOOTH TRAILING EDGE
RTE       = z[npoints_surface-2];
x0TE      = x[npoints_surface-2];
dzdx      = (z[npoints_surface-2]-z[npoints_surface-3])/(x[npoints_surface-2]-x[npoints_surface-3]);
deltax    = -z[npoints_surface-2]*dzdx;
x0        = x[npoints_surface-2] - (deltax) - c/2;
x0_ref    = x[npoints_surface-2] + (deltax);
R         = Sqrt((x[npoints_surface-2]-x0_ref)^2.0 + z[npoints_surface-2]^2);
Printf("%f", R);

// CIRCLE SEGMENTS
ncirc_disc = 10;  // number of lines used to discretize the tangent circle
p         = newp;
Point(p)  = {x0, 0.0, 0.0};
theta_a   = Atan2(z[npoints_surface-2], (x[npoints_surface-2]-x0_ref));
theta_b   = 2*Pi - theta_a;
dtheta    = (theta_b-theta_a)/(ncirc_disc);
Printf("%f", p);
For i In {0:ncirc_disc-1}
	theta = theta_a + i*dtheta;
	Point(p+i+1) = {x0 - R*Cos(theta), 0.0, R*Sin(theta)};
EndFor
Circle(2) = {npoints_surface-2, p, p+2};
For i In {2:ncirc_disc-1}
	Circle(1+i) = {p+i, p, p+i+1};
EndFor
Circle(1+ncirc_disc) = {p+ncirc_disc, p, npoints_surface-1};

// FARFIELD
p = newp;
R = 20.0;
Point(p)   = {0.0, 0.0, 0.0};
Point(p+1) = {R, 0.0, 0.0};
Point(p+2) = {0.0, 0.0, R};
Point(p+3) = {-R, 0.0, 0.0};
Point(p+4) = {0.0, 0.0, -R};
Circle(2+ncirc_disc) = {p+1, p, p+2};
Circle(3+ncirc_disc) = {p+2, p, p+3};
Circle(4+ncirc_disc) = {p+3, p, p+4};
Circle(5+ncirc_disc) = {p+4, p, p+1};
Curve Loop(1) = {2+ncirc_disc, 3+ncirc_disc, 4+ncirc_disc, 5+ncirc_disc};
Curve Loop(2) = {1:ncirc_disc+1};
Plane Surface(1) = {1, 2};

// BACKGROUND MESH
Fbackground = 2.0;
Field[0] = Box;
Field[0].VIn = Fbackground;
Field[0].VOut = Fbackground+1;
Field[0].XMax = R;
Field[0].XMin = -R;
Field[0].YMax = 2*c;
Field[0].YMin = -2*c;
Field[0].ZMax = R;
Field[0].ZMin = -R;
Background Field = 0;

// MESHING AROUND AIRFOIL
LcMin = 2E-4;
Field[1] = Distance;
Field[1].EdgesList = {1:ncirc_disc+1};
Field[1].NNodesByEdge = npoints_airfoil;
Field[2] = Threshold;
Field[2].DistMax = 2.0;
Field[2].DistMin = 0.0;
Field[2].IField = 1;
Field[2].LcMax = Fbackground;
Field[2].LcMin = LcMin;
Field[2].Sigmoid = 0;
Background Field = 2;

/*
Field[3] = BoundaryLayer;
Field[3].EdgesList = {1:ncirc_disc+1};
Field[3].hfar = LcMin;
Field[3].hwall_n = 2.0e-5;
Field[3].thickness = 1.5*LcMin;
BoundaryLayer Field = 3;
*/

// EXTRUDE MESH FOR 3D
Extrude {0, 1, 0} {
  Surface{1}; Layers{1}; 
}

// BOUNDARY CONDITIONS
Physical Surface("InletFixed", 101) = {35, 47, 43, 39};
Physical Surface("Symmetry_1", 102) = {1};
Physical Surface("Symmetry_2", 103) = {92};
Physical Surface("StickMoving", 104) = {91, 87, 83, 79, 75, 71, 67, 63, 59, 55, 51};
Physical Volume("FluidMesh", 100) = {1};

