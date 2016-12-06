//#include <boost/lambda/lambda.hpp>
//#include <iostream>
//#include <iterator>
//#include <algorithm>
//#include <vector>
//
////#include <boost/libs/geometry/index/text/rtree/test_rtree.hpp>

//#include <new> //For std::nothrow
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/geometries.hpp>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bgi::linear<4> Params;
typedef bg::model::point<double, 3, bg::cs::cartesian> point_t;
typedef bgi::rtree<point_t, Params> Rtree_t;

extern "C" {
	int first_test() {
		return 12;
	}

	Rtree_t * create_tree(void) {
		return new(std::nothrow) Rtree_t;
	}	

	void add_points(Rtree_t * tree_ptr, unsigned long number_points, unsigned int dimension, double *(points[3])) {
		std::cout << number_points << " " << dimension << std::endl;
		std::cout << (*points[1]) << std::endl;
	}

	void add_point(Rtree_t * tree_ptr) {
		tree_ptr->insert(point_t(1., 2., 3.));
	}

    double access_list(unsigned long index, double * list) {
//        std::cout << list << std::endl;
//        std::cout << list[index] << std::endl;
        return list[index];
    }

    double access_2d_list(unsigned long index1, unsigned long index2, double * list) {
        unsigned long index = index1 * 3 + index2;
//        std::cout << list << std::endl;
//        std::cout << list[index] << std::endl;
        return list[index];
    }

//	void * delete_tree(void *ptr) {
//		delete(std::nothrow) ptr;
//	}
//
//	int fill_tree()
//	{
//
//		Pt refpoint(0, 0, 4);
//
//		for ( int i=0; i<10; i++ )  rtree.insert(Pt(0, 0, i));
//
//		std::vector<Pt> result;
//		rtree.query(bgi::nearest(refpoint, 1), std::back_inserter(result));
//
//		return result[0].get<2>();
//
//	}

}


