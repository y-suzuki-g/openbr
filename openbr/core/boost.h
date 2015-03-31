#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include "openbr/openbr_plugin.h"
#include "ml.h"

namespace br
{

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
    CascadeBoostTree(Representation *rep) : rep(rep) {}
    virtual bool train( CvDTreeTrainData* trainData,
                        const CvMat* subsample_idx, CvBoost* ensemble );
    virtual float predict( const cv::Mat &img, bool isPrecalc = false ) const;

    float maxVal() const;

    void store( QDataStream &stream ) const;
    void load( QDataStream &stream );

    void freeOldData();
    void freeTree();

protected:
    CascadeBoostNode *simple_root;
    Representation *rep;
};

class CascadeBoost : public CvBoost
{
public:
    CascadeBoost();
    ~CascadeBoost();

    bool train(cv::Mat &_data, const cv::Mat &_labels,
                const CascadeBoostParams& _params, Representation *rep);
    virtual float predict(const cv::Mat &img , bool applyThreshold = true, bool isPrecalc = false) const;

    void freeTrees();
    float getThreshold() const { return threshold; }

    void store(QDataStream &stream) const;
    void load(Representation *rep, QDataStream &stream);

protected:
    virtual bool set_params(const CvBoostParams& _params);
    virtual bool isErrDesired();

    QList<CascadeBoostTree *> classifiers;
    cv::Mat trainData, labels;
    CvMat _oldData, _oldLabels; // these need to exist for stupid casting reasons
    float threshold;
    float normFactor;
    float minTAR, maxFAR;
};

} //namespace br

#endif
