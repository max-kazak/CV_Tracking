"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.S = 0.1 * np.eye(4)    # covariance

        self.Q = Q  # process noise
        self.R = R  # meas noise

        self.D = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])   # transition matrix

        self.M = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])   # measurement matrix

    def predict(self):
        pred_state = np.dot(self.D, self.state)
        pred_S = np.dot(np.dot(self.D, self.S), self.D.T) + self.Q

        self.state = pred_state
        self.S = pred_S

    def correct(self, meas_x, meas_y):
        K = np.dot(
                np.dot(self.S, self.M.T),
                np.linalg.inv(
                   np.dot(np.dot(self.M, self.S), self.M.T) + self.R
                )
            )
        Y = np.array([meas_x, meas_y])

        corr_state = self.state + np.dot(K, (Y - np.dot(self.M, self.state)))
        corr_S = np.dot((np.eye(4) - np.dot(K, self.M)), self.S)

        self.state = corr_state
        self.S = corr_S

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder

        self.template = template
        # self.template = frame[int(self.template_rect['y']):
        #                       int(self.template_rect['y'] + self.template_rect['h']),
        #                       int(self.template_rect['x']):
        #                       int(self.template_rect['x'] + self.template_rect['w'])]

        self.template = to_gray(self.template)
        self.template[self.template < 120] = 0

        self.frame = frame

        h, w = frame.shape[0], frame.shape[1]

        tx = self.template_rect.get('x', 0)
        ty = self.template_rect.get('y', 0)
        tw = self.template_rect.get('w', w)
        th = self.template_rect.get('h', h)

        # self.particles = np.array([
        #     np.random.randint(tx, tx+tw, self.num_particles),
        #     np.random.randint(ty, ty+th, self.num_particles)
        # ]).T

        self.particles = np.array([
            [int(tx + tw/2)] * self.num_particles,
            [int(ty + th/2)] * self.num_particles
        ]).T

        self.weights = np.array([1./self.num_particles] * self.num_particles)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        # th, tw = template.shape[0], template.shape[1]
        # mse = 1. * np.sum((template - frame_cutout) ** 2) / (th * tw)

        mse = np.sum((template.astype(np.float) - frame_cutout.astype(np.float))**2)
        mse /= np.float(template.shape[0] * template.shape[1])

        p = np.exp(-mse / (2 * self.sigma_exp))
        return p

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        inds = np.random.choice(self.num_particles, self.num_particles, p=self.get_weights())
        return self.particles[inds]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = to_gray(frame)

        # cv2.imshow('actual_frame', frame)

        h, w = frame.shape[0], frame.shape[1]

        new_particles = []
        new_weights = []

        for state in self.resample_particles():
            x, y = state

            # Dynamics
            new_x = x + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_y = y + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            if 0 < new_x < w - 1:
                x = new_x
            if 0 < new_y < h - 1:
                y = new_y
            # x = min(max(0, x), w - 1)
            # y = min(max(0, y), h-1)

            new_particles.append([x, y])

            # Weighting
            frame_cutout, template_cutout = calc_frame_cutout(x, y, frame, self.template)
            particle_weight = self.get_error_metric(template_cutout, frame_cutout)
            new_weights.append(particle_weight)

        new_weights = np.array(new_weights)
        weights_sum = np.sum(new_weights)
        if weights_sum > 0:
            self.weights = new_weights / weights_sum

        self.particles = np.array(new_particles)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        x_weighted_mean = int(x_weighted_mean)
        y_weighted_mean = int(y_weighted_mean)

        h, w = frame_in.shape[:2]
        th, tw = self.template.shape[:2]

        # frame_out = frame_in.copy()
        frame_out = frame_in

        # paint particles
        for i in range(self.num_particles):
            px, py = self.particles[i, 0], self.particles[i, 1]
            cv2.circle(frame_out, (px, py), 1, (255,255,255), -1)

        # paint border
        border_y_start = y_weighted_mean - th/2
        border_y_end = y_weighted_mean + th/2
        border_x_start = x_weighted_mean - tw/2
        border_x_end = x_weighted_mean + tw/2
        if th % 2 == 0: border_y_start += 1
        if tw % 2 == 0: border_x_start += 1

        cv2.rectangle(frame_out,
                      (max(0, border_x_start), max(0, border_y_start)),
                      (min(w-1, border_x_end), min(h-1, border_y_end)),
                      (255, 0, 0),
                      3)

        # paint uncertainty
        dist_weighted_mean = 0

        for i in range(self.num_particles):
            dist = calc_dist(x_weighted_mean, y_weighted_mean,
                             self.particles[i, 0], self.particles[i, 1])
            dist_weighted_mean += dist * self.weights[i]
        dist_weighted_mean = int(dist_weighted_mean)

        cv2.circle(frame_out, (x_weighted_mean, y_weighted_mean), dist_weighted_mean, (0, 255, 0), 3)

        return frame_out


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha', 0)  # required by the autograder

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        frame = to_gray(frame)

        # cv2.imshow('actual_frame', frame)

        h, w = frame.shape[0], frame.shape[1]

        new_particles = []
        new_weights = []

        best_match_weight = 0
        best_match = None

        for state in self.resample_particles():
            x, y = state

            # Dynamics
            new_x = x + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_y = y + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))

            if 0 < new_x < w - 1:
                x = new_x
            if 0 < new_y < h - 1:
                y = new_y

            new_particles.append([x, y])

            # Weighting
            frame_cutout, template_cutout = calc_frame_cutout(x, y, frame, self.template)
            particle_weight = self.get_error_metric(template_cutout, frame_cutout)
            new_weights.append(particle_weight)

            if particle_weight > best_match_weight and frame_cutout.shape == self.template.shape:
                best_match_weight = particle_weight
                best_match = frame_cutout

        new_weights = np.array(new_weights)
        weights_sum = np.sum(new_weights)
        if weights_sum > 0:
            self.weights = new_weights / weights_sum

        self.particles = np.array(new_particles)

        # Update template
        # cv2.imshow('best_match', best_match)

        if (best_match is not None) and (best_match.shape == self.template.shape):
            cur_template = self.template.astype(np.float32)
            cur_template = cv2.normalize(cur_template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #
            best_match = best_match.astype(np.float32)
            best_match = cv2.normalize(best_match, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            new_template = (best_match * self.alpha) + (cur_template * (1-self.alpha))

            new_template = cv2.normalize(new_template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            self.template = new_template.astype(np.uint8)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.scale_min = kwargs.get('scale_min', 50)
        self.scale_max = kwargs.get('scale_max', 100)
        self.scale_sigma = kwargs.get('scale_sigma', 3)
        self.scale_mean = kwargs.get('scale_mean', 0)

        scales = np.random.randint(self.scale_min, self.scale_max+1, self.num_particles)

        self.particles = np.insert(self.particles, 2, scales, axis=1)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = to_gray(frame)

        # cv2.imshow('actual_frame', frame)

        h, w = frame.shape[0], frame.shape[1]

        new_particles = []
        new_weights = []

        best_match_weight = 0
        best_match = None

        print 'scale_min: ', np.min(self.particles[:,2]), \
            'scale_max: ', np.max(self.particles[:,2]), \
            'scale_mean: ', np.mean(self.particles[:,2])

        for state in self.resample_particles():
            x, y, scale = state

            # Dynamics
            new_x = x + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_y = y + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_scale = scale + int(round(np.random.normal(self.scale_mean, self.scale_sigma, 1)[0]))

            if 0 < new_x < w - 1:
                x = new_x
            if 0 < new_y < h - 1:
                y = new_y
            if self.scale_min <= new_scale <= self.scale_max:
                scale = new_scale

            new_particles.append([x, y, scale])

            # Weighting
            particle_template = self.template.copy()
            particle_template = cv2.resize(particle_template, (0, 0), fx=scale/100., fy=scale/100.)
            frame_cutout, template_cutout = calc_frame_cutout(x, y, frame, particle_template)

            particle_weight = self.get_error_metric(template_cutout, frame_cutout)
            new_weights.append(particle_weight)

            if particle_weight > best_match_weight and frame_cutout.shape == self.template.shape:
                best_match_weight = particle_weight
                best_match = frame_cutout

        new_weights = np.array(new_weights)
        weights_sum = np.sum(new_weights)
        if weights_sum > 0:
            self.weights = new_weights / weights_sum

        self.particles = np.array(new_particles)

        # Update template
        # cv2.imshow('best_match', best_match)

        if (self.alpha != 0) and (best_match is not None) and (best_match.shape == self.template.shape):
            cur_template = self.template.astype(np.float32)
            cur_template = cv2.normalize(cur_template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #
            best_match = best_match.astype(np.float32)
            best_match = cv2.normalize(best_match, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            new_template = (best_match * self.alpha) + (cur_template * (1 - self.alpha))

            new_template = cv2.normalize(new_template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            self.template = new_template.astype(np.uint8)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0
        scale_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            scale_weighted_mean += self.particles[i, 2] * self.weights[i]

        # Complete the rest of the code as instructed.
        x_weighted_mean = int(x_weighted_mean)
        y_weighted_mean = int(y_weighted_mean)

        h, w = frame_in.shape[:2]
        th, tw = self.template.shape[:2]
        th = int(th * scale_weighted_mean / 100.)
        tw = int(tw * scale_weighted_mean / 100.)

        # frame_out = frame_in.copy()
        frame_out = frame_in

        # paint particles
        for i in range(self.num_particles):
            px, py = self.particles[i, 0], self.particles[i, 1]
            cv2.circle(frame_out, (px, py), 1, (255, 255, 255), -1)

        # paint border
        border_y_start = y_weighted_mean - th / 2
        border_y_end = y_weighted_mean + th / 2
        border_x_start = x_weighted_mean - tw / 2
        border_x_end = x_weighted_mean + tw / 2
        if th % 2 == 0: border_y_start += 1
        if tw % 2 == 0: border_x_start += 1

        cv2.rectangle(frame_out,
                      (max(0, border_x_start), max(0, border_y_start)),
                      (min(w - 1, border_x_end), min(h - 1, border_y_end)),
                      (255, 0, 0),
                      3)

        # paint uncertainty
        dist_weighted_mean = 0

        for i in range(self.num_particles):
            dist = calc_dist(x_weighted_mean, y_weighted_mean,
                             self.particles[i, 0], self.particles[i, 1])
            dist_weighted_mean += dist * self.weights[i]
        dist_weighted_mean = int(dist_weighted_mean)

        cv2.circle(frame_out, (x_weighted_mean, y_weighted_mean), dist_weighted_mean, (0, 255, 0), 3)

        return frame_out


class CustomPF(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(CustomPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.v_min = kwargs.get('v_min', -40)
        self.v_max = kwargs.get('v_max', +40)
        self.v_sigma = kwargs.get('v_sigma', 3)
        self.v_mean = kwargs.get('v_mean', 0)

        vs = np.random.randint(self.v_min, self.v_max + 1, self.num_particles)

        self.particles = np.insert(self.particles, 2, vs, axis=1)

        self.id = kwargs.get('id', 'noname')

        self.render_color = kwargs.get('render_color', (255, 0, 0))

        self.end_frame = kwargs.get('end_frame', -1)

    def process(self, frame):
        frame = to_gray(frame)

        # cv2.imshow('actual_frame', frame)

        h, w = frame.shape[0], frame.shape[1]

        new_particles = []
        new_weights = []

        print self.id + ": v=" + str(np.mean(self.particles[:,2]))

        for state in self.resample_particles():
            x, y, v = state

            # Dynamics
            new_x = x + v + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_y = y + int(round(np.random.normal(0, self.sigma_dyn, 1)[0]))
            new_v = v + int(round(np.random.normal(self.v_mean, self.v_sigma, 1)[0]))
            if 0 < new_x < w - 1:
                x = new_x
            if 0 < new_y < h - 1:
                y = new_y
            if self.v_min <= new_v <= self.v_max:
                v = new_v

            new_particles.append([x, y, v])

            # Weighting
            frame_cutout, template_cutout = calc_frame_cutout(x, y, frame, self.template)
            particle_weight = self.get_error_metric(template_cutout, frame_cutout)
            new_weights.append(particle_weight)

        new_weights = np.array(new_weights)
        weights_sum = np.sum(new_weights)
        if weights_sum > 0:
            self.weights = new_weights / weights_sum

        self.particles = np.array(new_particles)

    def render(self, frame_in):
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        x_weighted_mean = int(x_weighted_mean)
        y_weighted_mean = int(y_weighted_mean)

        h, w = frame_in.shape[:2]
        th, tw = self.template.shape[:2]

        # frame_out = frame_in.copy()
        frame_out = frame_in

        # paint particles
        for i in range(self.num_particles):
            px, py = self.particles[i, 0], self.particles[i, 1]
            cv2.circle(frame_out, (px, py), 1, self.render_color, -1)

        # paint border
        border_y_start = y_weighted_mean - th/2
        border_y_end = y_weighted_mean + th/2
        border_x_start = x_weighted_mean - tw/2
        border_x_end = x_weighted_mean + tw/2
        if th % 2 == 0: border_y_start += 1
        if tw % 2 == 0: border_x_start += 1

        cv2.rectangle(frame_out,
                      (max(0, border_x_start), max(0, border_y_start)),
                      (min(w-1, border_x_end), min(h-1, border_y_end)),
                      self.render_color,
                      3)

        # paint uncertainty
        dist_weighted_mean = 0

        for i in range(self.num_particles):
            dist = calc_dist(x_weighted_mean, y_weighted_mean,
                             self.particles[i, 0], self.particles[i, 1])
            dist_weighted_mean += dist * self.weights[i]
        dist_weighted_mean = int(dist_weighted_mean)

        cv2.circle(frame_out, (x_weighted_mean, y_weighted_mean), dist_weighted_mean, self.render_color, 3)

        return frame_out


# =========== helper functions ============
def to_gray(img):
    gray_image = img.copy()
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    # gray_image = gray_image.astype(np.float32)
    # gray_image *= np.array([.4, .4, .2])
    # gray_image = np.sum(gray_image, axis=2)
    # gray_image = gray_image.astype(np.uint8)
    return gray_image


def calc_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def calc_frame_cutout(x, y, frame, template):
    template_cutout = template

    h, w = frame.shape[0], frame.shape[1]
    th, tw = template.shape[0], template.shape[1]

    f_row_start, f_row_end = y - th / 2, y + th / 2
    f_col_start, f_col_end = x - tw / 2, x + tw / 2

    if th % 2 == 0: f_row_start += 1
    if tw % 2 == 0: f_col_start += 1

    if f_row_start < 0:
        template_cutout = template_cutout[-f_row_start:, :]
        f_row_start = 0
    elif f_row_end > h-1:
        template_cutout = template_cutout[:th - (f_row_end - (h-1)), :]
        f_row_end = h-1

    if f_col_start < 0:
        template_cutout = template_cutout[:, -f_col_start:]
        f_col_start = 0
    elif f_col_end > w-1:
        template_cutout = template_cutout[:, :tw - (f_col_end - (w-1))]
        f_col_end = w-1

    frame_cutout = frame[f_row_start:f_row_end+1, f_col_start:f_col_end+1]

    return frame_cutout, template_cutout
