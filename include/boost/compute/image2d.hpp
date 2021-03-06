//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://kylelutz.github.com/compute for more information.
//---------------------------------------------------------------------------//

#ifndef BOOST_COMPUTE_IMAGE2D_HPP
#define BOOST_COMPUTE_IMAGE2D_HPP

#include <vector>

#include <boost/throw_exception.hpp>

#include <boost/compute/cl.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/exception.hpp>
#include <boost/compute/image_format.hpp>
#include <boost/compute/memory_object.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/detail/get_object_info.hpp>

namespace boost {
namespace compute {

class image2d : public memory_object
{
public:
    /// Creates a null image2d object.
    image2d()
        : memory_object()
    {
    }

    /// Creates a new image2d object.
    ///
    /// \see_opencl_ref{clCreateImage}
    image2d(const context &context,
            cl_mem_flags flags,
            const image_format &format,
            size_t image_width,
            size_t image_height,
            size_t image_row_pitch = 0,
            void *host_ptr = 0)
    {
        cl_int error = 0;

        #ifdef CL_VERSION_1_2
        cl_image_desc desc;
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = image_width;
        desc.image_height = image_height;
        desc.image_depth = 1;
        desc.image_array_size = 0;
        desc.image_row_pitch = image_row_pitch;
        desc.image_slice_pitch = 0;
        desc.num_mip_levels = 0;
        desc.num_samples = 0;
        desc.buffer = 0;

        m_mem = clCreateImage(context,
                              flags,
                              format.get_format_ptr(),
                              &desc,
                              host_ptr,
                              &error);
        #else
        m_mem = clCreateImage2D(context,
                                flags,
                                format.get_format_ptr(),
                                image_width,
                                image_height,
                                image_row_pitch,
                                host_ptr,
                                &error);
        #endif

        if(!m_mem){
            BOOST_THROW_EXCEPTION(runtime_exception(error));
        }
    }

    /// Creates a new image2d as a copy of \p other.
    image2d(const image2d &other)
      : memory_object(other)
    {
    }

    /// Copies the image2d from \p other.
    image2d& operator=(const image2d &other)
    {
        memory_object::operator=(other);

        return *this;
    }

    /// Destroys the image2d object.
    ~image2d()
    {
    }

    /// Returns the format for the image.
    image_format get_format() const
    {
        return image_format(get_info<cl_image_format>(CL_IMAGE_FORMAT));
    }

    /// Returns the height of the image.
    size_t height() const
    {
        return get_info<size_t>(CL_IMAGE_HEIGHT);
    }

    /// Returns the width of the image.
    size_t width() const
    {
        return get_info<size_t>(CL_IMAGE_WIDTH);
    }

    /// Returns information about the image.
    ///
    /// \see_opencl_ref{clGetImageInfo}
    template<class T>
    T get_info(cl_image_info info) const
    {
        return detail::get_object_info<T>(clGetImageInfo, m_mem, info);
    }

    size_t get_pixel_count() const
    {
        return height() * width();
    }

    /// Returns the supported image formats for the context.
    ///
    /// \see_opencl_ref{clGetSupportedImageFormats}
    static std::vector<image_format> get_supported_formats(const context &context,
                                                           cl_mem_flags flags)
    {
        cl_uint count = 0;
        clGetSupportedImageFormats(context,
                                   flags,
                                   CL_MEM_OBJECT_IMAGE2D,
                                   0,
                                   0,
                                   &count);

        std::vector<cl_image_format> cl_formats(count);
        clGetSupportedImageFormats(context,
                                   flags,
                                   CL_MEM_OBJECT_IMAGE2D,
                                   count,
                                   &cl_formats[0],
                                   0);

        std::vector<image_format> formats;
        for(size_t i = 0; i < count; i++){
            formats.push_back(image_format(cl_formats[i]));
        }

        return formats;
    }
};

} // end compute namespace
} // end boost namespace

BOOST_COMPUTE_TYPE_NAME(boost::compute::image2d, image2d_t)

#endif // BOOST_COMPUTE_IMAGE2D_HPP
