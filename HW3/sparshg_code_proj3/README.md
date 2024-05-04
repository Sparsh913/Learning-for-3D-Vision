# Instructions to run the entire code

## Q0 attached in webpage

## Q1.3, Q1.4, Q1.5
use this command: python volume_rendering_main.py --config-name=box

## Q2.1, Q2.2, Q2.3
python volume_rendering_main.py --config-name=train_box

## Q3, Q4.1
python volume_rendering_main.py --config-name=nerf_lego
switch the view_dep bool to True for Q4.1

## Q5
python -m surface_rendering_main --config-name=torus_surface

## Q6
python -m surface_rendering_main --config-name=points_surface

## Q7
python -m surface_rendering_main --config-name=volsdf_surface

## Q8.2
python -m surface_rendering_main --config-name=volsdf_surface
randomly choose 10 training views -> changed the train_idx in `dataset.py`

# Q 8.3
python -m surface_rendering_main --config-name=volsdf_surface
added a naive implementation of the sdf to density from the given paper