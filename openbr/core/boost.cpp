#include <openbr/core/boost.h>
#include <openbr/core/opencvutils.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "cxmisc.h"

using namespace std;
using namespace cv;

namespace br
{

//----------------------------- CascadeBoostParams -------------------------------------------------

CascadeBoostParams::CascadeBoostParams() : minTAR( 0.995F), maxFAR( 0.5F ) {}

CascadeBoostParams::CascadeBoostParams(int _boostType, int _maxWeakCount, double _weightTrimRate, int _maxDepth, float _minTAR, float _maxFAR) :
    CvBoostParams( _boostType, _maxWeakCount, _weightTrimRate, _maxDepth, false, 0 )
{
    minTAR = _minTAR;
    maxFAR = _maxFAR;
}

void CascadeBoostParams::store(QDataStream &stream) const
{
    stream << boost_type;
    stream << minTAR;
    stream << maxFAR;
    stream << weight_trim_rate;
    stream << max_depth;
    stream << weak_count;
}

void CascadeBoostParams::load(QDataStream &stream)
{
    stream >> boost_type;
    stream >> minTAR;
    stream >> maxFAR;
    stream >> weight_trim_rate;
    stream >> max_depth;
    stream >> weak_count;
}

//-------------------------------- CascadeBoostTree ----------------------------------------

static void buildSimpleTree( CascadeBoostNode *node, CvDTreeNode *other_node )
{
    node->value = other_node->value;
    node->hasChildren = (other_node->left ? true : false);
    if (node->hasChildren) {
        node->feature_idx = other_node->split->var_idx;
        node->threshold = other_node->split->ord.c;

        node->left = new CascadeBoostNode, node->right = new CascadeBoostNode;
        buildSimpleTree(node->left, other_node->left);
        buildSimpleTree(node->right, other_node->right);
    }
}

bool CascadeBoostTree::train(CvDTreeTrainData *trainData, const CvMat *subsample_idx,
                                CvBoost *ensemble)
{
    if (!CvBoostTree::train( trainData, subsample_idx, ensemble ))
        return false;

    simple_root = new CascadeBoostNode;
    buildSimpleTree(simple_root, root);
    return true;
}

float CascadeBoostTree::predict(const cv::Mat &img , bool isPrecalc) const
{
    CascadeBoostNode *node = simple_root;
    if ( !node )
        CV_Error( CV_StsError, "tree has not been trained yet" );

    while ( node->hasChildren )
    {
        float val = isPrecalc ? img.at<float>(node->feature_idx) : rep->evaluate( img, QList<int>() << node->feature_idx ).at<float>(0);
        node = val <= node->threshold ? node->left : node->right;
    }
    return node->value;
}

static void storeNodeRecursive( CascadeBoostNode *node, QDataStream &stream )
{
    stream << node->value;
    stream << node->hasChildren; // has children
    if (node->hasChildren) {
        stream << node->feature_idx;
        stream << node->threshold;
        storeNodeRecursive(node->left, stream);
        storeNodeRecursive(node->right, stream);
    }
}

void CascadeBoostTree::store( QDataStream &stream ) const
{
    storeNodeRecursive( simple_root, stream );
}

static void loadNodeRecursive( CascadeBoostNode *node, QDataStream &stream)
{
    stream >> node->value;
    stream >> node->hasChildren;
    if (node->hasChildren) {
        stream >> node->feature_idx;
        stream >> node->threshold;

        node->left = new CascadeBoostNode, node->right = new CascadeBoostNode;
        loadNodeRecursive(node->left, stream);
        loadNodeRecursive(node->right, stream);
    }
}

void CascadeBoostTree::load(QDataStream &stream )
{
    simple_root = new CascadeBoostNode;
    loadNodeRecursive( simple_root, stream );
}

static void freeNodeRecursive( CascadeBoostNode *node )
{
    if (node->hasChildren) {
        freeNodeRecursive(node->left);
        freeNodeRecursive(node->right);
        delete node->left;
        delete node->right;
    }
    delete node;
}

void CascadeBoostTree::freeTree()
{
    freeNodeRecursive(simple_root);
}

void CascadeBoostTree::freeOldData()
{
    free_tree();
}

//----------------------------------- CascadeBoost --------------------------------------

bool CascadeBoost::train( Mat &_data, const Mat &_labels, const CascadeBoostParams& _params, Representation *rep )
{
    CV_Assert( !data );
    clear();

    trainData = _data;
    labels = _labels;

    _oldData = trainData;
    _oldLabels = labels;
    data = new CvDTreeTrainData(&_oldData, CV_ROW_SAMPLE, &_oldLabels, NULL, NULL, NULL, NULL, _params, true, true);

    set_params( _params );
    if ( (_params.boost_type == LOGIT) || (_params.boost_type == GENTLE) )
        data->do_responses_copy();

    update_weights( 0 );

    cout << "+----+---------+---------+" << endl;
    cout << "|  N |    HR   |    FA   |" << endl;
    cout << "+----+---------+---------+" << endl;

    do
    {
        CascadeBoostTree* tree = new CascadeBoostTree( rep );
        if( !tree->train( data, subsample_mask, this ) )
        {
            delete tree;
            break;
        }

        classifiers.append(tree);
        update_weights( tree );
        trim_weights();
        tree->freeOldData(); // releases old root after updating weights

        if( cvCountNonZero(subsample_mask) == 0 )
            break;
    }
    while( !isErrDesired() && (classifiers.size() < params.weak_count) );

    if(classifiers.empty()) {
        clear();
        return false;
    }

    data->free_train_data();
    delete data;

    return true;
}

float CascadeBoost::predict( const Mat &img, bool applyThreshold, bool isPrecalc ) const
{
    double sum = 0;
    foreach (const CascadeBoostTree *classifier, classifiers)
        sum += classifier->predict( img, isPrecalc );

    if (applyThreshold)
        return (float)sum - threshold;
    return (float)sum;
}

bool CascadeBoost::set_params( const CvBoostParams& _params )
{
    minTAR = ((CascadeBoostParams&)_params).minTAR;
    maxFAR = ((CascadeBoostParams&)_params).maxFAR;
    return ( ( minTAR > 0 ) && ( minTAR < 1) &&
        ( maxFAR > 0 ) && ( maxFAR < 1) &&
        CvBoost::set_params( _params ));
}

bool CascadeBoost::isErrDesired()
{
    int sampleCount = data->sample_count;
    QList<float> responses;

    for( int i = 0; i < sampleCount; i++ )
        if( labels.at<float>(i) == 1.0F )
            responses.append(predict( trainData.row(i), false, true));
    sort(responses.begin(), responses.end());

    int numPos = responses.size(), numNeg = sampleCount - numPos;

    int thresholdIdx = (int)((1.0F - minTAR) * numPos);
    threshold = responses[thresholdIdx];

    int numTrueAccepts = numPos - thresholdIdx;
    for( int i = thresholdIdx - 1; i >= 0; i--) // add pos values lower than the threshold that have the same response
        if ( responses[i] - threshold > -FLT_EPSILON )
            numTrueAccepts++;
    float TAR = ((float) numTrueAccepts) / ((float) numPos);

    int numFalseAccepts = 0;
    for( int i = 0; i < sampleCount; i++ )
        if( labels.at<float>(i) == 0.0F )
            if( predict( trainData.row(i), true, true) > -FLT_EPSILON)
                numFalseAccepts++;
    float FAR = ((float) numFalseAccepts) / ((float) numNeg);

    cout << "|"; cout.width(4); cout << right << classifiers.size();
    cout << "|"; cout.width(9); cout << right << TAR;
    cout << "|"; cout.width(9); cout << right << FAR;
    cout << "|" << endl;
    cout << "+----+---------+---------+" << endl;

    return FAR <= maxFAR;
}

void CascadeBoost::freeTrees()
{
    for (int i = 0; i < classifiers.size(); i++)
        classifiers[i]->freeTree();
}

void CascadeBoost::store( QDataStream &stream ) const
{
    ((CascadeBoostParams&)params).store( stream );

    stream << classifiers.size();
    stream << threshold;
    foreach (const CascadeBoostTree *classifier, classifiers)
        classifier->store(stream);
}

void CascadeBoost::load(Representation *rep, QDataStream &stream )
{
    // clear old data
    clear();
    classifiers.clear();

    ((CascadeBoostParams&)params).load( stream );

    int numTrees;
    stream >> numTrees;
    stream >> threshold;

    for (int i = 0; i < numTrees; i++) {
        CascadeBoostTree *classifier = new CascadeBoostTree( rep );
        classifier->load( stream );
        classifiers.append(classifier);
    }
}

}
