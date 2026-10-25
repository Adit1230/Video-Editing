"""Frame sampling regressions; no model or GPU dependencies."""
import contextlib
import io
import unittest
from run import clip_selection
from options import normalized_config,validate_options


class FrameSelection(unittest.TestCase):
    def config(self,**kw):
        c=dict(frames=17,start_frame=0,frame_stride=3,clip_seconds='auto',backend='cogvideox_2b')
        c.update(kw);return c

    def selection(self,c,fps,count):
        with contextlib.redirect_stdout(io.StringIO()):return clip_selection(c,fps,count)

    def test_native_opening_clip_too_short(self):
        with self.assertRaisesRegex(ValueError,'at least 1.134 seconds'):
            self.selection(self.config(clip_seconds=1.),15.,100)
        indices,fps=self.selection(self.config(),15.,100)
        self.assertEqual(len(indices),17);self.assertEqual(len(set(indices)),17)
        self.assertLess(indices[-1],30);self.assertEqual(fps,8.5)

    def test_automatic_low_fps_extends_for_unique_frames(self):
        for n in (5,9,17,49):
            indices,fps=self.selection(self.config(frames=n),2.,100)
            self.assertEqual(indices,list(range(n)))
            self.assertEqual(fps,2.)

    def test_automatic_shorter_video_with_enough_frames(self):
        indices,fps=self.selection(self.config(start_frame=4),15.,23)
        self.assertEqual(len(set(indices)),17)
        self.assertGreaterEqual(min(indices),4);self.assertLess(max(indices),23)
        self.assertAlmostEqual(17/fps,19/15)

    def test_insufficient_total_frames_reports_alternative(self):
        for seconds in ('auto',1.,3.):
            with self.assertRaisesRegex(ValueError,r'4\*k\+1'):
                self.selection(self.config(clip_seconds=seconds),15.,16)
        with self.assertRaisesRegex(ValueError,'0 remain'):
            self.selection(self.config(start_frame=100),15.,16)

    def test_explicit_duration_and_stride_unchanged(self):
        self.assertEqual(self.selection(self.config(frames=4,clip_seconds=2.),30.,100),([0,15,30,45],2.))
        self.assertEqual(self.selection(self.config(frames=4,clip_seconds=None),30.,100),([0,3,6,9],10.))
        indices,fps=self.selection(self.config(backend='sd14',frames=4),15.,100)
        self.assertEqual(indices,[0,3,7,11]);self.assertEqual(fps,4.)

    def test_configuration_validation(self):
        for seconds in ('auto',None,1.,2.):
            c=self.config(clip_seconds=seconds,size=320,steps=25,source_prompt='a cat',target_prompt='a dog')
            validate_options(normalized_config(c))
        for seconds in ('bad',0,-1,True,float('nan')):
            c=self.config(clip_seconds=seconds,size=320,steps=25,source_prompt='a cat',target_prompt='a dog')
            with self.assertRaises(ValueError):validate_options(normalized_config(c))


if __name__=='__main__':unittest.main()
