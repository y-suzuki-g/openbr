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
    int maxCatCount;
    float minTAR;
    float maxFAR;

    CascadeBoostParams();
    CascadeBoostParams( int _boostType, int _maxCatCount, float _minTAR, float _maxFAR,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
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

class CascadeBoostTree : public CvBoostTree
{
public:
    CascadeBoostTree() : maxCatCount(0) {}
    CascadeBoostTree(int _maxCatCount) : maxCatCount(_maxCatCount) {}

    virtual CvDTreeNode* predict( int sampleIdx ) const;

    void write(cv::FileStorage &fs);
    void read( const cv::FileNode &node, CvBoost* _ensemble, CvDTreeTrainData* _data );

protected:
    virtual void split_node_data( CvDTreeNode* n );

    int maxCatCount;
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
    float getThreshold() const { return threshold; }

    void save( const char *filename ) const;
    void write( cv::FileStorage &fs ) const;
    void load( const char *filename, const CascadeDataStorage *_storage, const CascadeBoostParams &_params);
    bool read( const cv::FileNode &node, const CascadeDataStorage* _storage,
               const CascadeBoostParams& _params );

protected:
    virtual bool set_params( const CvBoostParams& _params );
    virtual void update_weights( CvBoostTree* tree );
    virtual bool isErrDesired();

    float threshold;
    float minTAR, maxFAR;
};

} //namespace br

#endif
