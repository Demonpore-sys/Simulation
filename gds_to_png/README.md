I took the main Demonpore "actual-size" GDS file, and used klayout to remove everything BUT the 5th die (starting to count from the top-left of the wafer).


This was just to reduce complexity of the code for initial development.


I then used klayout to save-as the layer2 and layer4 in separate GDS files (again for clarity during development, since I'm still getting used to the gdsCAD API)

Then I can iterate through the boxes (slits) and points associated with each layer, and simply color those pixels in an output image.
