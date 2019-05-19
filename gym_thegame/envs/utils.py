import numpy as np


def draw_boundary(hero, state, args):
  """
  fill area out of arena boundary with boundary_color
  """
  hx, hy = hero.position
  hx, hy = hx / args.quantize, hy / args.quantize
  x_max, y_max = 5000 / args.quantize, 4000 / args.quantize
  up_bound = int(max(0, args.height / 2 - hy))
  left_bound = int(max(0, args.width / 2 - hx))
  down_bound = int(min(args.height, args.height / 2 + (y_max - hy)))
  right_bound = int(min(args.width, args.width / 2 + (x_max - hx)))
  state[0:up_bound, :, :] = args.boundary_color
  state[:, 0:left_bound, :] = args.boundary_color
  state[down_bound:args.width, :, :] = args.boundary_color
  state[:, right_bound:args.height, :] = args.boundary_color
  return state


def state_to_layer(game_state, args):
  """
  draw information into channels
  state channel define:
    0: total body damage (+alley, -enemy), boundary with
    1: entity health (include self)
    2: reward exp (+alley, -enemy)
  """
  hero, heroes, polygons, bullets = game_state

  # channel info type and rescale function
  channel_info = {
      'body_damage': (0, lambda v: v / 60),
      'health': (1, lambda v: v / 3000),
      'reward': (2, lambda v: v / 1000),
  }

  def draw(state, obj, value, channel_type):
    """
    draw `obj` relative to current hero position
    using rescaled channel info value
    """
    channel, rescale = channel_info[channel_type]
    value = np.clip(rescale(value), -1, 1)
    hx, hy = hero.position
    x, y = obj.position
    x, y, hx, hy = x / args.quantize, y / args.quantize, hx / args.quantize, hy / args.quantize
    radius_quantize = {0.5: 0, 1: 1, 1.25: 2, 1.5: 2}
    if obj.radius / args.quantize in radius_quantize:
      radius = radius_quantize[obj.radius / args.quantize]
    else:
      radius = math.ceil(obj.radius / args.quantize)
    x = round(x + args.width / 2 - hx)
    y = round(y + args.height / 2 - hy)
    lx, rx = int(max(0, x - radius)), int(min(args.width, x + radius + 1))
    ly, ry = int(max(0, y - radius)), int(min(args.height, y + radius + 1))
    for i in range(lx, rx):
      for j in range(ly, ry):
        if (i - x)**2 + (j - y)**2 <= radius**2:
          state[j, i, channel] = value

  # draw for every entity
  state = np.zeros((args.width, args.height, 3), dtype=np.float)

  enemy_bullets = [b for b in bullets if b.owner_id != hero.id]
  self_bullets = [b for b in bullets if b.owner_id == hero.id]
  for entity in [hero, *self_bullets]:
    draw(state, entity, entity.body_damage, 'body_damage')
  for entity in [*heroes, *polygons, *enemy_bullets]:
    draw(state, entity, -entity.body_damage, 'body_damage')
  for entity in [hero, *heroes, *polygons, *bullets]:
    draw(state, entity, entity.health, 'health')
  for entity in [*heroes, *polygons]:
    draw(state, entity, entity.rewarding_experience, 'reward')
  draw(state, hero, -entity.rewarding_experience, 'reward')

  state = draw_boundary(hero, state, args)
  return state, np.array(state * 255, dtype=np.uint8)


def state_to_rgb(game_state, args):
  """
  draw state with rgb or gray obv type
  """
  hero, heroes, polygons, bullets = game_state

  def interp(color_range, ratio):
    return (int(end - (end - start) * ratio)
            for start, end in zip(*color_range))

  def draw(state, obj, color_range):
    """
    draw `obj` relative to current hero position
    using color_range interpolated by current health
    """
    color = interp(color_range, obj.health / obj.max_health)
    hx, hy = hero.position
    x, y = obj.position
    x, y, hx, hy = x / args.quantize, y / args.quantize, hx / args.quantize, hy / args.quantize

    # radius adjustment
    radius_quantize = {0.5: 0, 1: 1, 1.25: 2, 1.5: 2}
    if obj.radius / args.quantize in radius_quantize:
      radius = radius_quantize[obj.radius / args.quantize]
    else:
      radius = math.ceil(obj.radius / args.quantize)

    # drawing range calculate
    x = round(x + args.width / 2 - hx)
    y = round(y + args.height / 2 - hy)
    lx, rx = int(max(0, x - radius)), int(min(args.width, x + radius + 1))
    ly, ry = int(max(0, y - radius)), int(min(args.height, y + radius + 1))
    # drawing value
    for channel, value in enumerate(color):
      for i in range(lx, rx):
        for j in range(ly, ry):
          if (i - x)**2 + (j - y)**2 <= radius**2:
            state[j, i, channel] = value

  if args.obv_type == 'rgb':
    bg_color = 255
    #
    # color range: (start rgb, end rgb)
    #
    # hero, own bullets: green
    hero_color = ((16, 79, 15), (119, 226, 118))
    # polygons: blue
    polygon_color = {
        3: ((33, 47, 104), (96, 115, 196)),
        4: ((33, 23, 84), (112, 98, 188)),
        5: ((23, 63, 89), (89, 148, 186)),
    }
    # heroes, other bullets: red
    other_color = ((114, 11, 11), (239, 103, 103))
  else:  # gray obv
    bg_color = args.bg_color
    hero_color = ((255, 255, 255), (50, 50, 50))
    polygon_color = {
        3: ((130, 130, 130), (0, 0, 0)),
        4: ((80, 80, 80), (0, 0, 0)),
        5: ((200, 200, 200), (0, 0, 0)),
    }
    other_color = ((180, 180, 180), (0, 0, 0))

  # draw for every entity
  state = np.full((args.width, args.height, 3), bg_color, dtype=np.float)
  for polygon in polygons:
    draw(state, polygon, polygon_color[polygon.edges])
  for bullet in bullets:
    if bullet.owner_id == hero.id:
      draw(state, bullet, hero_color)
    else:
      draw(state, bullet, other_color)
  for hero_ in heroes:
    draw(state, hero_, other_color)
  draw(state, hero, hero_color)

  state = draw_boundary(hero, state, args)
  return state / 255, np.array(state, dtype=np.uint8)


def state_to_gray(game_state, args):
  """
  draw state to gray obv
  """
  state_rgb, img = state_to_rgb(game_state, args)
  return state_rgb[:, :, 2:], img


get_to_state_fn = {
    'rgb': state_to_rgb,
    'layer': state_to_layer,
    'gray': state_to_gray
}
