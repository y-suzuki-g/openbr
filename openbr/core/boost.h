#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include "openbr/openbr_plugin.h"
#include "ml.h"

namespace br
{

struct CascadeBoostParams : CvBoostParams
{
    int maxCatCount;
    float minTAR;
    float maxFAR;

    CascadeBoostParams();
    CascadeBoostParams( int _boostType, int _maxCatCount, float _minTAR, float _maxFAR,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
    virtual ~CascadeBoostParams() {}
    void store(QDataStream &stream) const;
};

struct CascadeDataStorage
{
    CascadeDataStorage() {}
    CascadeDataStorage(int numFeatures, int numSamples);

    void setImage(const cv::Mat &sample, float label, int idx);

    int numSamples()  const { return data.cols; }
    int numPos()      const { return cv::countNonZero(labels); }
    int numNeg()      const { return data.cols - cv::countNonZero(labels); }
    int numFeatures() const { return data.rows; }
    float getResponse(int featureIdx, int sampleIdx) const { return data.at<float>(featureIdx, sampleIdx); }
    float getLabel(int sampleIdx) const { return labels.at<float>(sampleIdx); }

    cv::Mat data;
    cv::Mat labels;
};

struct CascadeBoostTrainData : CvDTreeTrainData
{
    CascadeBoostTrainData(CascadeDataStorage *_storage,
                             int _numSamples,
                             const CvDTreeParams& _params = CvDTreeParams() );
    void sort();

    virtual CvDTreeNode* subsample_data( const CvMat* _subsample_idx );

    virtual const int* get_class_labels( CvDTreeNode* n, int* labelsBuf );
    virtual const int* get_cv_labels( CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_sample_indices( CvDTreeNode* n, int* indicesBuf );

    virtual void get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
                                  const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf );
    virtual const int* get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf );
    virtual float getVarValue( int vi, int si );

    CascadeDataStorage *storage;
    CvMat _resp; // for casting
};

class CascadeBoostTree : public CvBoostTree
{
public:
    virtual CvDTreeNode* predict( int sampleIdx ) const;
    void store(QDataStream &stream) const;

protected:
    virtual void split_node_data( CvDTreeNode* n );
};

class CascadeBoost : public CvBoost
{
public:
    virtual bool train( CascadeDataStorage *_storage,
                        int _numSamples,
                        CascadeBoostParams& _params );
    virtual float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }
    void store(QDataStream &stream) const;

protected:
    virtual bool set_params( const CvBoostParams& _params );
    virtual void update_weights( CvBoostTree* tree );
    virtual bool isErrDesired();

    float threshold;
    float minTAR, maxFAR;
};

} //namespace cv

#endif
