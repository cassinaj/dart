#ifndef HOST_ONLY_MODEL_H
#define HOST_ONLY_MODEL_H

#include "model/model.h"
//#include <Eigen/Eigen>
#include <map>
#include <vector>

#include <GL/glew.h>

#include <vector_types.h>

#include "geometry/SE3.h"
#include "geometry/geometry.h"
#include "geometry/grid_3d.h"
#include "mesh/mesh.h"
#include "pose/pose.h"

namespace dart
{
class HostOnlyModel : public Model
{
  public:
    HostOnlyModel();
    ~HostOnlyModel();

    // model setup
    void addGeometry(int frame,
                     GeomType geometryType,
                     std::string,
                     std::string sy,
                     std::string sz,
                     std::string tx,
                     std::string ty,
                     std::string tz,
                     std::string rx,
                     std::string ry,
                     std::string rz,
                     unsigned char red,
                     unsigned char green,
                     unsigned char blue,
                     const std::string meshFilename = "");
    void addGeometry(int frame,
                     GeomType geometryType,
                     float sx,
                     float sy,
                     float sz,
                     float tx = 0,
                     float ty = 0,
                     float tz = 0,
                     float rx = 0,
                     float ry = 0,
                     float rz = 0,
                     unsigned char red = 0xff,
                     unsigned char green = 0xff,
                     unsigned char blue = 0xff,
                     const std::string meshFilename = "");

    int addFrame(int parent,
                 JointType type,
                 std::string posX,
                 std::string posY,
                 std::string posZ,
                 std::string orX,
                 std::string orY,
                 std::string orZ,
                 std::string axisX,
                 std::string axisY,
                 std::string axisZ,
                 std::string jointMin,
                 std::string jointMax,
                 std::string jointName,
                 std::string mimicJoint,
                 std::string mimicMultiplier,
                 std::string mimicOffset);

    void computeStructure();
    void voxelize(float resolution,
                  float padding = 0.0f,
                  std::string cacheFile = "");
    void voxelize2(float resolution, float padding, std::string cacheFile = "");

    void setJointLimits(const int joint, const float min, const float max);

    void setGeometryColor(int geomNumber, unsigned char color[3]);
    void setGeometryColor(int geomNumber,
                          unsigned char red,
                          unsigned char green,
                          unsigned char blue);

    // model queries
    bool hasMimicJoints() const { return mimicJointMap.size() > 0; }
    int getNumMimicJoints() const { return mimicJointMap.size(); }
    bool isMimicJoint(int jointIndex) const
    {
        return mimicJointMap.find(jointIndex) != mimicJointMap.end();
    }

    int getMimicJointSource(int jointIndex) const
    {
        if (!isMimicJoint(jointIndex))
        {
            std::cerr << "Joint " << jointIndex << " is not a mimic joint."
                      << std::endl;
            exit(1);
        }

        return mimicJointMap.find(jointIndex)->second;
    }
    float getMimicMultiplier(int jointIndex) const
    {
        if (!isMimicJoint(jointIndex))
        {
            std::cerr << "Joint " << jointIndex << " is not a mimic joint."
                      << std::endl;
            exit(1);
        }

        return mimicMultiplierMap.find(jointIndex)->second;
    }

    float getMimicOffset(int jointIndex) const
    {
        if (!isMimicJoint(jointIndex))
        {
            std::cerr << "Joint " << jointIndex << " is not a mimic joint."
                      << std::endl;
            exit(1);
        }

        return mimicOffsetMap.find(jointIndex)->second;
    }
    uint getNumFrames() const { return _nFrames; }
    int getFrameParent(const int frame) const { return _parents[frame]; }
    int getFrameNumChildren(const int frame) const
    {
        return _children[frame].size();
    }
    const int* getFrameChildren(const int frame) const
    {
        return _children[frame].data();
    }

    int getFrameJoint(int frame) const
    {
        for (int i = 0; i < getNumJoints(); i++)
            if (getJointFrame(i) == frame) return i;
        return -1;
    }

    const int* getDependencies() const { return _dependencies.data(); }
    int getDependency(const int frame, const int joint) const
    {
        return _dependencies[frame * getNumJoints() + joint];
    }

    const SE3& getTransformFrameToModel(const int frame) const
    {
        return _T_mf[frame];
    }
    const SE3& getTransformModelToFrame(const int frame) const
    {
        return _T_fm[frame];
    }
    const SE3* getTransformsFrameToModel() const { return _T_mf.data(); }
    const SE3* getTransformsModelToFrame() const { return _T_fm.data(); }
    int getFrameNumGeoms(const int frame) const
    {
        return _frameGeoms[frame].size();
    }
    const int* getFrameGeoms(const int frame) const
    {
        return _frameGeoms[frame].data();
    }

    int getGeometryFrame(const int geom_number) const
    {
        for (int i = 0; i < _nFrames; i++)
            for (int j = 0; j < getFrameNumGeoms(i); j++)
                if (getFrameGeoms(i)[j] == geom_number) return i;
        return -1;
    }
    uchar3 getGeometryColor(const int geomNumber) const
    {
        return _geomColors[geomNumber];
    }

    const std::string& getGeometryMeshFilename(const int geomNumber) const
    {
        return _meshFilenames.at(geomNumber);
    }

    // joint queries
    uint getNumJoints() const { return _jointNames.size(); }
    int getJointFrame(const int joint) const { return joint + 1; }
    JointType getJointType(const int joint) const { return _jointTypes[joint]; }
    float3 getJointAxis(const int joint) const { return _axes[joint]; }
    float3& getJointPosition(const int joint) { return _positions[joint]; }
    float3& getJointOrientation(const int joint)
    {
        return _orientations[joint];
    }
    const float3& getJointPosition(const int joint) const
    {
        return _positions[joint];
    }
    const float3& getJointOrientation(const int joint) const
    {
        return _orientations[joint];
    }

    uint getNumSdfs() const { return _sdfs.size(); }
    const Grid3D<float>& getSdf(const int sdfNum) const
    {
        return _sdfs[sdfNum];
    }
    const Grid3D<float>* getSdfs() const { return _sdfs.data(); }
    uint getSdfFrameNumber(const int sdfNum) const
    {
        return _sdfFrames[sdfNum];
    }
    uchar3 getSdfColor(const int sdfNum) const { return _sdfColors[sdfNum]; }
    void setVoxelGrid(Grid3D<float>& grid, const int link);

    // set parameters of DH model
    void setArticulation(const float* pose);
    void setPose(const Pose& pose);
    void setParam(const int link, const int param, const float value);
    void setGeometryScale(const int geom, const float3 scale);
    void setGeometryTransform(const int geom, const SE3& T);
    void setLinkParent(const int link, const int parent);

    void addSizeParam(std::string name, float value)
    {
        _sizeParams[name] = value;
    }
    const std::map<std::string, float>& getSizeParams() const
    {
        return _sizeParams;
    }
    void setSizeParam(const std::string param, const float value)
    {
        std::map<std::string, float>::const_iterator it =
            _sizeParams.find(param);
        if (it != _sizeParams.end())
        {
            _sizeParams[param] = value;
        }
    }
    float getSizeParam(const std::string param) const
    {
        std::map<std::string, float>::const_iterator it =
            _sizeParams.find(param);
        if (it != _sizeParams.end())
        {
            return it->second;
        }
        return 0.0;
    }

    void setSdfColor(const int sdfNum, const uchar3 color)
    {
        _sdfColors[sdfNum] = color;
    }
    void setGeometryColor(const int geomNum, const uchar3 color)
    {
        _geomColors[geomNum] = color;
    }

    const SE3 getTransformJointAxisToParent(const int joint) const
    {
        return _T_pf[joint];
    }

  protected:
    int _nFrames;

    std::vector<int> _parents;
    std::vector<std::vector<int> > _children;
    std::vector<JointType> _jointTypes;
    std::vector<SE3> _T_mf;
    std::vector<SE3> _T_fm;
    std::vector<SE3> _T_pf;
    std::map<int, std::string> _meshFilenames;

    std::vector<int> _sdfFrames;

    std::vector<uchar3> _geomColors;
    std::vector<uchar3> _sdfColors;

    std::vector<Grid3D<float> > _sdfs;

    std::vector<int> _dependencies;

    std::map<std::string, float> _sizeParams;

    std::vector<float3> _axes;
    std::vector<float3> _positions;
    std::vector<float3> _orientations;
    std::vector<std::vector<std::string> > _frameParamExpressions;
    std::vector<std::vector<std::string> > _geomParamExpressions;

    std::map<std::string, int> jointNameIndexMap;
    std::vector<std::string> mimicJoints;
    std::map<int, int> mimicJointMap;
    std::map<int, float> mimicMultiplierMap;
    std::map<int, float> mimicOffsetMap;

    void voxelizeFrame(Grid3D<float>& vg,
                       const int frame,
                       const float fg,
                       const float bg,
                       const float resolution,
                       const float padding,
                       const float Rc = 1.0);

    float evaluateExpression(const std::string& expression);
};
}

#endif  // HOST_ONLY_MODEL_H
