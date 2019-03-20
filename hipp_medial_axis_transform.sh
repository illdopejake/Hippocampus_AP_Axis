### This script was written by Claude Lepage and edited by Jacob Vogel.

### The script relies on MINC Toolkit (version 1.0.08) 
### (RRID:SCR\_014138; http://bic-mni.github.io/\#MINC-Tool-Kit)


# input file is called blob.nii

# convert blob to minc

### nii2mnc blob.nii blob.mnc

# resample blob at 0.5mm in int16 (int16 is needed for mincchamfer to work)
# NOTE: autocrop will cause a small shift (half voxel size). Should fix this.

autocrop -clobber -isostep 0.5 blob.mnc /tmp/template.mnc

mincresample -quiet -clobber -like /tmp/template.mnc -nearest -unsigned -short blob.mnc blob_half.mnc 
#rm -f /tmp/template.mnc

# distance map 10mm away from blob.

mincchamfer -max_dist 10.0 blob_half.mnc blob_chamfer_out.mnc

# threshold distance map at 5mm, then create a distance map inside
# the 5mm region around the initial blob (this will be smooth).

minccalc -quiet -clob -unsigned -short -expr 'A[0]>5.0' blob_chamfer_out.mnc blob_chamfer5.mnc

mincchamfer -max_dist 20 blob_chamfer5.mnc blob_chamfer_in.mnc
mincblur -quiet -clob -no_apodize -fwhm 2.0 blob_chamfer_in.mnc blob_chamfer_in

#rm -f blob_chamfer_out.mnc blob_chamfer5.mnc blob_chamfer_in.mnc

# compute derivatives of the inner chamfer map.

mincmorph -clob -convolve -kernel dx.kernel blob_chamfer_in_blur.mnc dx.mnc
mincmorph -clob -convolve -kernel dy.kernel blob_chamfer_in_blur.mnc dy.mnc
mincmorph -clob -convolve -kernel dz.kernel blob_chamfer_in_blur.mnc dz.mnc
minccalc -quiet -clob -expr 'sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2])' dx.mnc dy.mnc dz.mnc blob_grad.mnc
#rm -f dx.mnc dy.mnc dz.mnc blob_chamfer_in_blur.mnc

# the magnitude of the derivative at the center line will be small.
# Elsewhere, the derivative is about 0.50 (distance map on voxels of 0.50mm).
# Keep it inside the original blob (removes edge effects).

minccalc -quiet -clob -expr 'if(A[0]>0.5&&A[1]<=0.25){1}else{0}' blob_half.mnc blob_grad.mnc blob_line.mnc
#rm blob_grad.mnc blob_half.mnc

mincskel blob_line.mnc blob_line_skel.mnc
#rm blob_line.mnc

mincchamfer -max_dist 10 blob_line_skel.mnc blob_midline.mnc
#rm blob_line_skel.mnc

# MOVE BACK TO 1mm and NIFTI
mincresample -quiet -clobber -like blob.mnc -nearest -unsigned -short blob_line_skel.mnc blob_line_skel_1mm.mnc
mnc2nii blob_line_skel_1mm.mnc blob_line_skel_1mm_2nii.nii

mincresample -quiet -clobber -like blob.mnc -nearest -unsigned -short blob_midline.mnc blob_midline_1mm.mnc
mnc2nii blob_midline_1mm.mnc blob_midline_1mm_2nii.nii
