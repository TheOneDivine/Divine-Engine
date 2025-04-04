#ifndef AABB_H
#define AABB_H

#include "interval.hpp"

class aabb {
public:
   interval x, y, z;

   aabb() {} // The default AABB is empty, since intervals are empty by default.

   aabb(const interval& x, const interval& y, const interval& z) : x(x), y(y), z(z) {}

   aabb(const glm::vec3& a, const glm::vec3& b) {
      // Treat the two points a and b as extrema for the bounding box, so we don't require a
      // particular minimum/maximum coordinate order.

      x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
      y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
      z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
   }

   const interval& axis_interval(int n) const {
      if (n == 1) return y;
      if (n == 2) return z;
      return x;
   }
};

#endif