#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <utility>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  is_initialized = true;

  std::normal_distribution<double> normal_distribution_x(x, std[0]);
  std::normal_distribution<double> normal_distribution_y(y, std[1]);
  std::normal_distribution<double> normal_distribution_theta(theta, std[2]);

  std::default_random_engine engine;
  num_particles = 10;
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i + 1;
    particle.x = normal_distribution_x(engine);
    particle.y = normal_distribution_y(engine);
    particle.theta = normal_distribution_theta(engine);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
}

void
ParticleFilter::prediction(double delta_t, double std_pos[], double velocity,
                           double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine random_engine;

  for (auto &particle : particles) {
    const double x_0 = particle.x;
    const double y_0 = particle.y;
    const double theta_0 = particle.theta;

    double x_f, y_f, theta_f;
    if (std::abs(yaw_rate) > 0.0001) {
      const double c_0 = velocity / yaw_rate;
      const double delta_theta = yaw_rate * delta_t;
      theta_f = theta_0 + delta_theta;
      x_f = x_0 + c_0 * (std::sin(theta_f) - std::sin(theta_0));
      y_f = y_0 + c_0 * (std::cos(theta_0) - std::cos(theta_f));
    } else {
      const double c0 = velocity * delta_t;
      x_f = x_0 + c0 * std::cos(theta_0);
      y_f = y_0 + c0 * std::sin(theta_0);
      theta_f = theta_0;
    }

    std::normal_distribution<double> normal_distribution_x(x_f, std_pos[0]);
    std::normal_distribution<double> normal_distribution_y(y_f, std_pos[1]);
    std::normal_distribution<double> normal_distribution_theta(theta_f,
                                                               std_pos[2]);

    particle.x = normal_distribution_x(random_engine);
    particle.y = normal_distribution_y(random_engine);
    particle.theta = normal_distribution_theta(random_engine);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (auto &observation : observations) {
    std::vector<double> distances(predicted.size());
    auto pred_dist = [&observation](const LandmarkObs &pred) {
        return dist(pred.x, pred.y, observation.x, observation.y);
    };
    std::transform(predicted.begin(), predicted.end(), distances.begin(),
                   pred_dist);
    const auto distance_min = std::min_element(distances.begin(),
                                               distances.end());
    const auto index_distance_min = std::distance(distances.begin(),
                                                  distance_min);
    observation.x -= predicted[index_distance_min].x;
    observation.y -= predicted[index_distance_min].y;
    observation.id = predicted[index_distance_min].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  weights.clear();
  for (auto &particle : particles) {
    std::vector<LandmarkObs> obstacles_predicted;

    for (auto &item : map_landmarks.landmark_list) {
      if (dist(item.x_f, item.y_f, particle.x, particle.y) < sensor_range) {
        LandmarkObs obstacle{};
        obstacle.x = item.x_f;
        obstacle.y = item.y_f;
        obstacle.id = item.id_i;
        obstacles_predicted.push_back(obstacle);
      }
    }

    auto observation_to_mapped_obs = [&particle](const LandmarkObs &obs) {
        LandmarkObs rval{};
        const double ct = std::cos(particle.theta);
        const double st = std::sin(particle.theta);
        rval.x = particle.x + (ct * obs.x) - (st * obs.y);
        rval.y = particle.y + (st * obs.x) + (ct * obs.y);
        rval.id = obs.id;
        return rval;
    };

    std::vector<LandmarkObs> obstacles_mapped(observations.size());
    std::transform(observations.begin(), observations.end(),
                   obstacles_mapped.begin(),
                   observation_to_mapped_obs);

//    std::vector<double> debug_seen_x(observations.size());
//    std::transform(obstacles_mapped.begin(), obstacles_mapped.end(), debug_seen_x.begin(),
//                   [](const LandmarkObs &obs) { return obs.x; });
//    std::vector<double> debug_seen_y(observations.size());
//    std::transform(obstacles_mapped.begin(), obstacles_mapped.end(), debug_seen_y.begin(),
//                   [](const LandmarkObs &obs) { return obs.y; });
    dataAssociation(obstacles_predicted, obstacles_mapped);
//    std::vector<int> debug_associations(observations.size());
//    std::transform(obstacles_mapped.begin(), obstacles_mapped.end(), debug_associations.begin(),
//                   [](const LandmarkObs &obs) { return obs.id; });

    const double sig_x = std_landmark[0];
    const double sig_y = std_landmark[1];
    const double c1 = 2.0 * sig_x * sig_x;
    const double c2 = 2.0 * sig_y * sig_y;
    const double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

    double weight = 1.0;

    for (auto &obs : obstacles_mapped) {
      const double dx = obs.x;
      const double dy = obs.y;
      const double z = (dx * dx / c1) + (dy * dy / c2);
      const double w = gauss_norm * exp(-z);
      weight *= w;
    }
    particle.weight = weight;
    weights.push_back(weight);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::default_random_engine randomEngine;
  std::discrete_distribution<> discrete_distribution1(weights.begin(),
                                                      weights.end());
  std::vector<Particle> particles_new;
  for (size_t i = 0; particles.size() > i; ++i) {
    const Particle &src = particles[discrete_distribution1(randomEngine)];
    particles_new.push_back(src);
  }
  particles.clear();
  particles.insert(particles.end(), particles_new.begin(), particles_new.end());
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = std::move(associations);
  particle.sense_x = std::move(sense_x);
  particle.sense_y = std::move(sense_y);

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
