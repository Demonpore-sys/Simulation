I took the main Demonpore "actual-size" GDS file, and used klayout to remove everything BUT the 5th die (starting to count from the top-left of the wafer).


This was just to reduce complexity of the code for initial development.


I then used klayout to save-as the layer2 and layer4 in separate GDS files (again for clarity during development, since I'm still getting used to the gdsCAD API)

Then I can iterate through the boxes (slits) and points associated with each layer, and simply color those pixels in an output image.

# References
* Beyond nanopore sizing: improving solid-state single-molecule sensing performance, lifetime, and analyte scope for omics by targeting surface chemistry during fabrication.pdf
```
place  this  unclogging behavior in  context, single-stranded  polynucleotides typically translocate
through SiNx nanopores at an average speed of ~1nucleotide/μs (at 150mV voltage bias).
To  resolve individual bases, it is estimated that  this  translocation  rate  should  be ~1 nucleotide/ms or slower.
At this desired rate, to sequence the 6 billion base pairs long diploid mammalian genome in 50,000 bases long ssDNA fragments
(after denaturing the dsDNA) with only a single translocation pass would require, without multiplexing,
a single pore to be open for >2000 hours.
Even with an array of >500 nanopore channels, it would require ~48 hours.
Even though  the  scope  of  this  work  is  not  DNA  sequencing,  the  stable CT-CDB open-pore  currents,
such  as  the  ~2.5  hours  of  current  trace  shown  in  Figure  S3,  should greatly  benefit  SSN-based efforts for sequencing
(the same pore was used for > 8 hours of experimentation yielding ~48,500 events over multiple dsDNA concentrations
 and applied voltages—over multiple experiments—and was still open and stable when decommissioned).
This open-pore current stability may also become a key positive aspect where a set of comparative experiments are
expected to be conducted using  the  same  nanopore  to  minimize intraand interpore  size  variations.
In  this  work,  for nanopores  fabricated  using  the CT-CDB  protocol,  we  show  (steady)  representative
continuous current traces as shown in Figure 2a (and zoomed in image in Figure 2b) that are ~1800 seconds long—the
longest continuous trace in the literature to the best of our knowledge—pertaining to dsDNA  translocating  through
a  ~3.4nm  diameter  pore. We  attempted  to  translocate  dsDNA through  a  similar  sizedpore  fabricated  from  the
 CDB protocol  butw ere met  with  continuous analyte-sticking which eventually lead to irreversible pore-clogging
 before any substantial number of events could be collected.
 The signal characteristics using CT-CDB pores are well-behaved in comparison to the CDB standard.
```

```
Calibration  curve  (inter-event  frequency  vs  dsDNA  concentration) constructed by adding
 1kb dsDNA (4M LiCl buffered at pH~7) in ~5nM increments to ~5nm diameter nanopores fabricated from
  the CDB protocol (magenta) and CT-CDB protocol (black).
   Each dsDNA aliquot of (d) was run for at least 900 seconds and each data point represents
    at least ~750 (CDB) and ~3800 (CT-CDB) events.
     Data were obtained using an applied voltage of +200mV, 250kHz of sampling frequency and 100kHz of low-pass filtering.
```


```
Events were classified as current perturbations at-least five times the standard deviation of the baseline current (𝐼0).
Each  event  was  then characterized in terms of amplitude (𝐼), duration and change in conductance (𝛥𝐺=𝐼0−𝐼𝑉).
```


