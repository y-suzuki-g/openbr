#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include "openbr/openbr_plugin.h"
#include "ml.h"

namespace br
{

struct CascadeDataStorage
{
    CascadeDataStorage() {}
    CascadeDataStorage(Representation *_rep, int numSamples);

    void setImage(const cv::Mat &sample, float label, int idx);
    void freeTrainData();

    int numSamples()  const { return data.cols; }
    int numFeatures() const { return rep->numFeatures(); }
    float response(int featureIdx, int sampleIdx) const;
    float label(int sampleIdx) const { return labels.at<float>(sampleIdx); }

    Representation *rep;
    cv::Mat data;
    cv::Mat labels;
};

struct CascadeBoostParams : CvBoostParams
{
    float minTAR;
    float maxFAR;

    CascadeBoostParams();
    CascadeBoostParams(int _boostType, int _maxWeakCount, double _weightTrimRate, int _maxDepth, float _minTAR, float _maxFAR );
    virtual ~CascadeBoostParams() {}
    void store(QDataStream &stream) const;
    void load(QDataStream &stream);
};

struct CascadeBoostTrainData : CvDTreeTrainData
{
    CascadeBoostTrainData( const CascadeDataStorage* _storage,
                             const CvDTreeParams& _params );
    CascadeBoostTrainData( const CascadeDataStorage* _storage,
                             int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                             const CvDTreeParams& _params = CvDTreeParams() );
    void precalculate();

    virtual CvDTreeNode* subsample_data( const CvMat* _subsample_idx );

    virtual const int* get_class_labels( CvDTreeNode* n, int* labelsBuf );
    virtual const int* get_cv_labels( CvDTreeNode* n, int* labelsBuf);
    virtual const int* get_sample_indices( CvDTreeNode* n, int* indicesBuf );

    virtual void get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
                                  const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf );
    virtual const int* get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf );
    virtual float getVarValue( int vi, int si );
    virtual void free_train_data();

    const CascadeDataStorage* storage;
    cv::Mat valCache; // precalculated feature values (CV_32FC1)
    CvMat _resp; // for casting
    int numPrecalcVal, numPrecalcIdx;
};

struct CascadeBoostNode
{
    float threshold;
    int feature_idx;
    float value;
    bool hasChildren;

    CascadeBoostNode *left;
    CascadeBoostNode *right;
};

class CascadeBoostTree : public CvBoostTree
{
public:
    virtual bool train( CvDTreeTrainData* trainData,
                        const CvMat* subsample_idx, CvBoost* ensemble );
    virtual float predict( int sampleIdx ) const;

    void store( QDataStream &stream ) const;
    void load( CvDTreeTrainData *_data, QDataStream &stream );

    void freeTree();

protected:
    virtual void split_node_data( CvDTreeNode* n );

    CascadeBoostNode *simple_root;
};

class CascadeBoost : public CvBoost
{
public:
    virtual bool train(const CascadeDataStorage *_storage,
                        int _numSamples,
                        int _precalcValBufSize, int _precalcIdxBufSize,
                        const CascadeBoostParams &_params=CascadeBoostParams() );
    virtual float predict(int sampleIdx , bool applyThreshold) const;

    void freeTrainData() { data->free_train_data(); }
    void freeTrees();
    float getThreshold() const { return threshold; }

    void store( QDataStream &stream ) const;
    void load( const CascadeDataStorage *_storage, CascadeBoostParams &_params, QDataStream &stream );

protected:
    virtual bool set_params( const CvBoostParams& _params );
    virtual void update_weights( CvBoostTree* tree );
    virtual bool isErrDesired();

    QList<CascadeBoostTree *> classifiers;
    float threshold;
    float minTAR, maxFAR;
};

} //namespace br

#endif
