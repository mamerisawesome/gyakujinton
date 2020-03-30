# Gyaku Jinton

OpenCV wrapper to handle shapes and images.

## Pre-requisites

* Python 3.6.1 or higher
* OpenCV

I suggest to check OpenCV's [tutorials](https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.htmlv) on installation to help you.

## Programmatic

### Drawing a polygon on image

Given an image path, by identifying points in the image, you can draw line based on given point.

If `color` is added as argument, it will also draw those lines by that color. The color is in RGB format.

If `output_path` is not defined, the application will create an application window with the modified image.

```python
from gyakujinton import draw_on_image
draw_on_image(
    image_path="/path/to/file.filetype",
    output_path="/path/to/output-file.filetype",  #optional
    points=[[INT, INT], ..., [INT, INT]],  # points on a 2D plane
    color=(0, 0, 0)  # in RGB; optional
)
```

## Command-line Interface

The application also allows executions through the CLI.

### Drawing a polygon on image (CLI)

The example below gets the image based on the path given and will draw a polygon based on the input points.

```bash
gyakujinton draw_on_image /path/to/file.filetype --points 100,100 200,100 200,200 100,200
```

We can also define an output path by adding the argument `-o` or `--output_path` followed by the file path.

```bash
gyakujinton draw_on_image /path/to/file.filetype --points 100,100 200,100 200,200 100,200 --output_path /path/to/output-file.filetype
```

## Sample

To test out the application, we'll be using an [image by Samantha Gades](https://unsplash.com/photos/BlIhVfXbi9s) taken from Unsplash. We have this beautiful and simple original image.

![Original Image](https://raw.githubusercontent.com/mamerisawesome/gyakujinton/master/sample/samantha-gades-unsplash.jpg)

We can draw a polygon near the clock by doing the following in the command-line:

```bash
gyakujinton draw_on_image sample/samantha-gades-unsplash.jpg --points 150,150 150,250 250,250 250,150
```

![Modified Image](https://raw.githubusercontent.com/mamerisawesome/gyakujinton/master/sample/output.jpg)

Again, we are not restricted with the polygon that we want to create. It all depends on where we put the points and how they are ordered when lines are to be drawn.

```bash
gyakujinton draw_on_image sample/samantha-gades-unsplash.jpg --points 100,100 100,150 180,200 250,150 250,100 180,50 -o output-2.jpg
```

![Modified Image 2](https://raw.githubusercontent.com/mamerisawesome/gyakujinton/master/sample/output-2.jpg)

## Name Inspiration

![Ohnoki's Dust Release](https://vignette.wikia.nocookie.net/naruto/images/2/20/Dust_Release.png/revision/latest/scale-to-width-down/1000?cb=20150123214535)

> Source: naruto.fandom.com/wiki

When thinking of a name for the app, the first thing that came into mind is Ohnoki's [Particle Style (or Dust Release) Atomic Dismantling Jutsu](https://naruto.fandom.com/wiki/Dust_Release:_Detachment_of_the_Primitive_World_Technique) from [Naruto](https://www.viz.com/naruto) which is a technique that has a sphere in the center contained by a geometric object. In the series, dust release is called `Jinton` which was chosen due to how amazed I am on the shapes happening.

Now, with the points above, the technique is used to dismantle atoms to dust. `Gyaku` (or reverse as taught to me by [Google Translate](https://translate.google.com/?gs_lcp=CgZwc3ktYWIQAzIICCEQFhAdEB46BQgAEIMBOgIIADoECAAQCjoFCAAQxAI6CAgAEBYQChAeOgYIABAWEB46BAghEApQ8ghYqDFgxjJoAnAAeACAAZ4BiAG7HpIBBDAuMzGYAQCgAQGqAQdnd3Mtd2l6&uact=5&um=1&ie=UTF-8&hl=en&client=tw-ob#auto/ja/reverse)) was added to signifying making of shapes and images rather than dismantling them.

Hence, the app name `Gyaku Jinton`.

## Author

Almer Mendoza
