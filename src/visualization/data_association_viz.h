#ifndef DATA_ASSOCIATION_VIZ_H
#define DATA_ASSOCIATION_VIZ_H

#include <vector_types.h>

namespace dart {

void colorDataAssociation(uchar3 * coloredAssociation,
                          const int * dataAssociation,
                          const uchar3 * colors,
                          const int width,
                          const int height,
                          const uchar3 unassociatedColor = make_uchar3(0,0,0));

void colorDataAssociationMultiModel(uchar3 * coloredAssociation,
                                    const int * dataAssociation,
                                    const uchar3 * * colors,
                                    const int width,
                                    const int height,
                                    const uchar3 unassociatedColor = make_uchar3(0,0,0));

void getIndicesFromDataAssociationMultiModel(int2 * modelSDFIndices,
                                             const int * dataAssociation,
                                             const int width,
                                             const int height,
                                             const int2 unassociatedVals = make_int2(-1,-1));

}

#endif // DATA_ASSOCIATION_VIZ_H
