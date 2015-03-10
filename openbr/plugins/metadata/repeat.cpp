#include <openbr/plugins/openbr_internal.h>

namespace br
{

class RepeatTransform : public UntrainableTransform
{
    Q_OBJECT

    Q_PROPERTY(br::Transform* transform READ get_transform WRITE set_transform RESET reset_transform STORED false)
    Q_PROPERTY(int n READ get_n WRITE set_n RESET reset_n STORED false)
    BR_PROPERTY(br::Transform*, transform, NULL)
    BR_PROPERTY(int, n, 1)

    void project(const Template &src, Template &dst) const
    {
        for (int i = 0; i < n; i++) {
            transform->project(src, dst);
        }
    }
};

BR_REGISTER(Transform, RepeatTransform)

} // namespace br

#include "metadata/repeat.moc"
