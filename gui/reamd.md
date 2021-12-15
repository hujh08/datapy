GUI for data

# Image Marker
Navigate and mark images one by one

## Brief of underlying logic
This marker consists of 3 parts -- graphical interface, images iterator and record system. **Graphical interface** is front end for the user. It displays image (and other informatin, e.g. image name, num of marked images), and respond to event triggered by the user, like key press. **Images iterator** fetches an image and passes to graphical interface. In the iterator, image is picked out in an sequatial or random order. **Record system** notes down user's mark for current image, for example, simple label (good or bad), or other comments (To support later). This system is divided to local and global part. *Local record* just takes notes for current image, and after image changes or marker quits, these notes are merged to correponding entry in *global record*, in which marks for al images are stored.

## Statistic
After several loops of mark, one image may have more than one labels, for example `{lab0: n0, lab1: n1, ...}`. Question is how to choose one as final label.

## Examples
