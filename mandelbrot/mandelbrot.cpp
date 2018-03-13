// Mandelbrot set example
// Adam Sampson <a.sampson@abertay.ac.uk>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <iostream>
#include <amp.h>
#include <amp_math.h>
#include <array>

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::complex;
using std::cout;
using std::endl;
using std::ofstream;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_clock;


// The size of the image to generate.
const int WIDTH = 640;
const int HEIGHT = 480;
const int RES = WIDTH * HEIGHT;

// The number of times to iterate before we assume that a point isn't in the
// Mandelbrot set.
// (You may need to turn this up if you zoom further into the set.)
const int MAX_ITERATIONS = 4000;

// The image data.
// Each pixel is represented as 0xRRGGBB.
//uint32_t image[HEIGHT][WIDTH];
std::array<uint32_t, RES> image;
//std::vector<std::vector<uint32_t>> image(HEIGHT, std::vector<uint32_t>(WIDTH));


// using our own structure as Complex function not available in the Concurrency namespace
struct Complex1 {
	float x;
	float y;
};
Complex1 c_add(Complex1 c1, Complex1 c2) restrict(cpu, amp) // restrict keyword - able to execute this function on the GPU and CPU
{
	Complex1 tmp;
	float a = c1.x;
	float b = c1.y;
	float c = c2.x;
	float d = c2.y;
	tmp.x = a + c;
	tmp.y = b + d;
	return tmp;
} // c_add
float c_abs(Complex1 c) restrict(cpu, amp)
{
	return concurrency::fast_math::sqrt(c.x*c.x + c.y*c.y);
} // c_abs
Complex1 c_mul(Complex1 c1, Complex1 c2) restrict(cpu, amp)
{
	Complex1 tmp;
	float a = c1.x;
	float b = c1.y;
	float c = c2.x;
	float d = c2.y;
	tmp.x = a*c - b*d;
	tmp.y = b*c + a*d;
	return tmp;
} // c_mul


// Write the image to a TGA file with the given name.
// Format specification: http://www.gamers.org/dEngine/quake3/TGA.txt
void write_tga(const char *filename)
{
	ofstream outfile(filename, ofstream::binary);

	uint8_t header[18] = {
		0, // no image ID
		0, // no colour map
		2, // uncompressed 24-bit image
		0, 0, 0, 0, 0, // empty colour map specification
		0, 0, // X origin
		0, 0, // Y origin
		WIDTH & 0xFF, (WIDTH >> 8) & 0xFF, // width
		HEIGHT & 0xFF, (HEIGHT >> 8) & 0xFF, // height
		24, // bits per pixel
		0, // image descriptor
	};
	outfile.write((const char *)header, 18);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			int loc = (y * WIDTH) + x;

			uint8_t pixel[3] = {
				image[loc] & 0xFF, // blue channel
				(image[loc] >> 8) & 0xFF, // green channel
				(image[loc] >> 16) & 0xFF, // red channel
			};
			outfile.write((const char *)pixel, 3);
		}
	}

	outfile.close();
	if (!outfile)
	{
		// An error has occurred at some point since we opened the file.
		cout << "Error writing to " << filename << endl;
		exit(1);
	}
}


// Render the Mandelbrot set into the image array.
// The parameters specify the region on the complex plane to plot.
void compute_mandelbrot(float left, float right, float top, float bottom, std::array<uint32_t, RES>& img)
{
	concurrency::extent<2> e(HEIGHT, WIDTH);
	concurrency::array_view<uint32_t, 2> smol_brot(e, img);
	smol_brot.discard_data();

	try {
		concurrency::parallel_for_each(smol_brot.extent, [=](concurrency::index<2> idx) restrict(amp) {
			int x = idx[1];
			int y = idx[0];

			// Work out the point in the complex plane that
			// corresponds to this pixel in the output image.
			Complex1 c = { left + (x * (right - left) / WIDTH), top + (y * (bottom - top) / HEIGHT) };

			// Start off z at (0, 0).
			Complex1 z = {0.0, 0.0};

			// Iterate z = z^2 + c until z moves more than 2 units
			// away from (0, 0), or we've iterated too many times.
			int iterations = 0;
			while (c_abs(z) < 2.0 && iterations < MAX_ITERATIONS)
			{
				z = c_add(c_mul(z, z), c);

				++iterations;
			}

			if (iterations == MAX_ITERATIONS)
			{
				// z didn't escape from the circle.
				// This point is in the Mandelbrot set.
				smol_brot[idx] = 0x000000; //black
			}
			else
			{
				// z escaped within less than MAX_ITERATIONS
				// iterations. This point isn't in the set.
				//smol_brot[1][x] = 0xFFFFFF; // white
				int temp = iterations / 2;

				smol_brot[idx] = (temp << 16) | (temp << 8) | temp; //grayscale
			}

		});
	}
	catch (const std::exception& ex) {
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
}


int main(int argc, char *argv[])
{
	cout << "Please wait..." << endl;

	// Start timing
	the_clock::time_point start = the_clock::now();

	// This shows the whole set.
	//compute_mandelbrot(-2.0, 1.0, 1.125, -1.125, image);

	// This zooms in on an interesting bit of detail.
	compute_mandelbrot(-0.751085, -0.734975, 0.118378, 0.134488, image);

	// Stop timing
	the_clock::time_point end = the_clock::now();

	// Compute the difference between the two times in milliseconds
	auto time_taken = duration_cast<milliseconds>(end - start).count();
	cout << "Computing the Mandelbrot set took " << time_taken << " ms." << endl;

	write_tga("output.tga");

	std::cin.ignore();

	return 0;
}
