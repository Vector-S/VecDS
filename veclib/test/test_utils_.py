from veclib.utils import *





def test_load_save_json():
    cfg = {
        'para1' : 123,
        'para2' : '34'

    }
    save_json(cfg,'../tmp/sample_cfg.cfg')
    assert  cfg == load_json('../tmp/sample.cfg')

def test_load_save_obj():
    obj = {
        'a':2,
        'b':1
    }
    filename = '../tmp/sample_obj'
    save_obj(obj,filename)
    assert obj==load_obj(filename)
if __name__=='__main__':
    test_load_save_json()
    test_load_save_obj()