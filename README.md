# uoc2022_tfm

https://brain-development.org/ixi-dataset/

preparation: download t1 and demographic information. Try to view the content with an appropiate viewer (preliminary work)

demographic information is useful for splitting between training and testing sets.

NIFTI format - don't need to know it - but we need a viewer i.e. ITK-SNAP

figure out how to load NIFTI files into python platforms and extract slices of the brain with same axes always - isotropic - slices are the same resolution. 

if you load 1 image it will have 3-d information and you can get different slices of the brain moving up&down the brain - 

working in 2-d or 3-d, 2-d is recommended as we'd need less resources. With 2d we'd be able to get around 200 slices of the brain.

possible variations:
- instead of reconstruction use variational autoencoders or GAN to produce artificial MRI brain images


Default project: design and train an autoencoder to reconstruct T1-weighted MRI images (remove artefacts? fill-in blanked out-regions?)

Alternative: do instead a variational autoencoder or GAN to synthesise artificial brain MRI images

Alternative 2: Not do an autoencoder but do a classification network to predict the gender from the MRI (this might need to be a 3D network)

Use existing MRI autoncoders as a starting point - we don't need to come up with something new, but we might try different architectures and find the optimal architecture and maybe modify an existing one to make it better.

Check AdrianArnaiz github repo (previous student doing the project)
