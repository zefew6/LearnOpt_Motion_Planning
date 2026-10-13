import numpy as np
from types import SimpleNamespace
from uav_ac.YOPO.audit import audit_candidates


def test_audit_finds_component_gradients_and_detects_wrong_backward():
    class Oracle:
        bad=False
        def evaluate(self,obs,z):
            cost=(z*z).sum(1)
            return SimpleNamespace(costs=cost,free_gradients=2*z if not self.bad else np.zeros_like(z),
                components={'progress':(z[:,8:16]**2).sum(1),'other':cost-(z[:,8:16]**2).sum(1)})
    z=np.ones((2,34));z[:,-2:]=1.
    oracle=Oracle();result=audit_candidates(None,oracle,z,[0])
    assert result['passed'] and result['candidates'][0]['max_relative_gradient_error']<1e-5
    assert result['candidates'][0]['component_terminal_gradient_norm']['progress']>0
    oracle.bad=True
    assert not audit_candidates(None,oracle,z,[0])['passed']


def test_promotion_requires_both_dense_feasibility_and_real_reaching():
    from uav_ac.YOPO.acceptance import gate
    r={'families':{'free':{'samples':1,'selected_dense_valid_fraction':1.}},'online':{'free':{'success':True,'collision':False,'maximum_handoff_body_rate_difference':.1}}}
    assert gate(r)
    r['online']['free']['success']=False;assert not gate(r)
    r['online']['free']['success']=True;r['families']['free']['selected_dense_valid_fraction']=.8;assert not gate(r)


def test_gate_rejects_missing_online_family_and_empty_validation():
    from uav_ac.YOPO.acceptance import gate
    r={'families':{'free':{'samples':1,'selected_dense_valid_fraction':1.},'wall':{'samples':1,'selected_dense_valid_fraction':1.}},'online':{'free':{'success':True,'collision':False,'maximum_handoff_body_rate_difference':.1}}}
    assert not gate(r)
    del r['families']['wall'];r['families']['free']['samples']=0;assert not gate(r)


def test_online_config_does_not_require_pilot_checkpoint(tmp_path):
    from uav_ac.YOPO.acceptance import online_config
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'metadata.json').write_text('{}');(checkpoint/'weights.pt').write_bytes(b'fixture')
    c=online_config(checkpoint,'free',10.)
    assert c['yopo']['checkpoint']==str(checkpoint)
    assert c['scene'].endswith('yopo_stage_free.xml') and not c['visualize']


def test_neural_audit_rejects_detached_terminal_positions():
    import torch,runpy
    from uav_ac.YOPO.audit import audit_backward
    from uav_ac.planning.end2end.primitive_network import PrimitivePlannerNet
    from uav_ac.planning.end2end.primitive import encode_primitive,primitive_anchors
    from uav_ac.planning.end2end.encoding import Normalization
    class Oracle:
        def evaluate(self,obs,z):return SimpleNamespace(costs=(z*z).sum(1),free_gradients=2*z)
    o=runpy.run_path('tests/unit/planning/end2end/test_rolling.py')['moving_observation']()
    n=Normalization((-2.8,-2.2,-2.4,-2.4),(2.8,2.2,2.4,2.4));net=PrimitivePlannerNet(n,width=4).double().eval()
    e=encode_primitive(o,n);state=torch.tensor(e.state,dtype=torch.float64);anchors=torch.tensor(primitive_anchors(o,n)[None],dtype=torch.float64)
    def run():
        output=net(torch.tensor(e.depth,dtype=torch.float64),state,anchors)
        return audit_backward(output,o,Oracle(),net,state,anchors,[13])
    assert run()['passed']
    original=net.decode
    def detached(raw,s,a):
        z=original(raw,s,a);return torch.cat((z[:,:,:8],z[:,:,8:11].detach(),z[:,:,11:]),dim=-1)
    net.decode=detached
    assert not run()['passed']
