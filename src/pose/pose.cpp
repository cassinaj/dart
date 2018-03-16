#include "pose.h"
#include "model/host_only_model.h"

namespace dart
{
Pose::Pose(PoseReduction* reduction)
    : _reduction(reduction),
      _fullArticulation(new float[reduction->getFullDimensions()])
{
    if (reduction->isNull())
    {
        _reducedArticulation = _fullArticulation;
    }
    else
    {
        _reducedArticulation = new float[reduction->getReducedDimensions()];
    }
}

Pose::Pose(const Pose& pose)
    : _reduction(pose._reduction),
      _fullArticulation(new float[pose._reduction->getFullDimensions()])
{
    if (pose._fullArticulation == pose._reducedArticulation)
    {
        _reducedArticulation = _fullArticulation;
    }
    else
    {
        _reducedArticulation =
            new float[pose._reduction->getReducedDimensions()];
    }
    memcpy(_reducedArticulation,
           pose._reducedArticulation,
           (pose._reduction->getReducedDimensions()) * sizeof(float));
    _T_cm = pose._T_cm;
    _T_mc = pose._T_mc;
}

Pose::~Pose()
{
    delete[] _fullArticulation;
    if (_reducedArticulation != _fullArticulation)
    {
        delete[] _reducedArticulation;
    }
}

Pose& Pose::operator=(const Pose& rhs)
{
    memcpy(_reducedArticulation,
           rhs._reducedArticulation,
           getReducedArticulatedDimensions() * sizeof(float));
    memcpy(_fullArticulation,
           rhs._fullArticulation,
           getArticulatedDimensions() * sizeof(float));
    _T_mc = rhs._T_mc;
    _T_cm = rhs._T_cm;
    return *this;
}

LinearPoseReduction::LinearPoseReduction(const int fullDims, const int redDims)
    : PoseReduction(fullDims, redDims),
      _A(fullDims * redDims),
      _b(new float[fullDims])
{
}

LinearPoseReduction::LinearPoseReduction(const int fullDims,
                                         const int redDims,
                                         float* A,
                                         float* b,
                                         float* mins,
                                         float* maxs,
                                         std::string* names)
    : PoseReduction(fullDims, redDims),
      _A(fullDims * redDims),
      _b(new float[fullDims])
{
    init(A, b, mins, maxs, names);
}

void LinearPoseReduction::init(
    float* A, float* b, float* mins, float* maxs, std::string* names)
{
    memcpy(_b, b, _fullDims * sizeof(float));
    memcpy(_A.hostPtr(), A, _fullDims * _redDims * sizeof(float));
    _A.syncHostToDevice();
    memcpy(_mins.data(), mins, _redDims * sizeof(float));
    memcpy(_maxs.data(), maxs, _redDims * sizeof(float));
    for (int i = 0; i < _redDims; ++i)
    {
        _names[i] = names[i];
    }
}

void LinearPoseReduction::projectReducedToFull(const float* reduced,
                                               float* full) const
{
    for (int f = 0; f < _fullDims; ++f)
    {
        full[f] = _b[f];
        for (int r = 0; r < _redDims; ++r)
        {
            full[f] += reduced[r] * _A.hostPtr()[r + f * _redDims];
        }
    }
}

ParamMapPoseReduction::ParamMapPoseReduction(const int fullDims,
                                             const int redDims,
                                             const int* mapping,
                                             float* mins,
                                             float* maxs,
                                             std::string* names)
    : LinearPoseReduction(fullDims, redDims), _mapping(fullDims)
{
    std::vector<float> b(fullDims, 0.f);
    std::vector<float> A(fullDims * redDims, 0.f);
    for (int f = 0; f < fullDims; ++f)
    {
        const int r = mapping[f];
        A[r + f * redDims] = 1;
    }

    init(A.data(), b.data(), mins, maxs, names);

    memcpy(_mapping.hostPtr(), mapping, fullDims * sizeof(float));
    _mapping.syncHostToDevice();
}

void ParamMapPoseReduction::projectReducedToFull(const float* reduced,
                                                 float* full) const
{
    for (int f = 0; f < _fullDims; ++f)
    {
        full[f] = reduced[_mapping[f]];
    }
}

LinearPoseReduction* LinearPoseReduction::CreateFromModel(
    const HostOnlyModel* model)
{
    int fullDimensions = model->getNumJoints();
    int numMimicJoints = model->getNumMimicJoints();
    int reducedDimensions = fullDimensions - numMimicJoints;
    auto reduction = new LinearPoseReduction(fullDimensions, reducedDimensions);

    float* A = new float[fullDimensions * reducedDimensions];
    float* b = new float[fullDimensions];
    float* mins = new float[reducedDimensions];
    float* maxs = new float[reducedDimensions];
    std::vector<float> jointMins;
    std::vector<float> jointMaxs;
    std::vector<std::string> jointNames;

    auto determineReducedIndex = [&](int jointIndex) {
        int index = 0;
        if (model->isMimicJoint(jointIndex))
        {
            // Get reference index
            int sourceJointIndex = model->getMimicJointSource(jointIndex);
            for (int i = 0; i < sourceJointIndex; ++i)
            {
                if (!model->isMimicJoint(i))
                {
                    index++;
                }
            }
        }
        else
        {
            for (int i = 0; i < jointIndex; ++i)
            {
                if (!model->isMimicJoint(i))
                {
                    index++;
                }
            }
        }
        return index;
    };

    for (int f = 0; f < fullDimensions; ++f)
    {
        for (int r = 0; r < reducedDimensions; ++r)
        {
            A[f * reducedDimensions + r] = 0;
        }
        float factor = 1.f;
        float bias = 0.f;
        if (model->isMimicJoint(f))
        {
            factor = model->getMimicMultiplier(f);
            bias = model->getMimicOffset(f);
        }
        int reducedIndex = determineReducedIndex(f);
        if (reducedIndex > fullDimensions)
        {
            std::cout << "Reduced joint index is out bounds (Larger than the "
                         "total state dimension) ... "
                      << std::endl;
            exit(1);
        }
        A[f * reducedDimensions + reducedIndex] = factor;
        b[f] = bias;

        if (!model->isMimicJoint(f))
        {
            jointMins.push_back(model->getJointMin(f));
            jointMaxs.push_back(model->getJointMax(f));
            jointNames.push_back(model->getJointName(f));
        }
    }

    reduction->init(
        A, b, jointMins.data(), jointMaxs.data(), jointNames.data());

    delete[] A;
    delete[] b;
    delete[] mins;
    delete[] maxs;

    return reduction;
}
}
