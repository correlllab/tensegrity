"""White-background 'paper mode' rendering helpers for figures."""
import re
import mujoco
import numpy as np

SKYBOX = ('<texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" '
          'width="256" height="256"/>')

def paper_xml(xml):
    """Light floor + white sky + brighter light for print figures."""
    xml = xml.replace('rgb1="0.2 0.25 0.3" rgb2="0.3 0.35 0.4"',
                      'rgb1="0.88 0.89 0.90" rgb2="0.96 0.96 0.97"')
    xml = xml.replace('<asset>', '<asset>\n    ' + SKYBOX)
    xml = xml.replace('<light pos="0 0 3" dir="0 0 -1" directional="true"/>',
                      '<light pos="0 0 3" dir="0 0 -1" directional="true" '
                      'diffuse="0.9 0.9 0.9" ambient="0.35 0.35 0.35"/>')
    return xml

def load_paper_model(path):
    m = mujoco.MjModel.from_xml_string(paper_xml(open(path).read()))
    m.vis.global_.offwidth, m.vis.global_.offheight = 2000, 1600
    return m

def render(m, d, cam_kw, w=1000, h=800):
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.azimuth = cam_kw.get("az", 130)
    cam.distance = cam_kw.get("dist", 2.3)
    cam.elevation = cam_kw.get("elev", -8)
    cam.lookat[:] = cam_kw.get("look", [0, 0, 0.85])
    r.update_scene(d, cam)
    out = r.render().copy()
    r.close()
    return out
