import random
import time
import pygame
import sys
import numpy as np
from collections import deque
from pygame.locals import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

FPS = 25
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
BOXSIZE = 20
BOARDWIDTH = 10
BOARDHEIGHT = 20
BLANK = 0

MOVESIDEWAYSFREQ = 0.15
MOVEDOWNFREQ = 0.1

XMARGIN = int((WINDOWWIDTH - BOARDWIDTH * BOXSIZE) / 2)
TOPMARGIN = WINDOWHEIGHT - (BOARDHEIGHT * BOXSIZE) - 5

WHITE = (255, 255, 255)
GRAY = (185, 185, 185)
BLACK = (0, 0, 0)
RED = (155, 0, 0)
LIGHTRED = (175, 20, 20)
GREEN = (0, 155, 0)
LIGHTGREEN = (20, 175, 20)
BLUE = (0, 0, 155)
LIGHTBLUE = (20, 20, 175)
YELLOW = (155, 155, 0)
LIGHTYELLOW = (175, 175, 20)

BORDERCOLOR = BLUE
BGCOLOR = BLACK
TEXTCOLOR = WHITE
TEXTSHADOWCOLOR = GRAY
COLORS = (BLUE, GREEN, RED, YELLOW)
LIGHTCOLORS = (LIGHTBLUE, LIGHTGREEN, LIGHTRED, LIGHTYELLOW)
assert len(COLORS) == len(LIGHTCOLORS)

TEMPLATEWIDTH = 5
TEMPLATEHEIGHT = 5

S_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '..OO.',
                     '.OO..',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..OO.',
                     '...O.',
                     '.....']]

Z_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '.OO..',
                     '..OO.',
                     '.....'],
                    ['.....',
                     '..O..',
                     '.OO..',
                     '.O...',
                     '.....']]

I_SHAPE_TEMPLATE = [['..O..',
                     '..O..',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     'OOOO.',
                     '.....',
                     '.....']]

O_SHAPE_TEMPLATE = [['.....',
                     '.....',
                     '.OO..',
                     '.OO..',
                     '.....']]

J_SHAPE_TEMPLATE = [['.....',
                     '.O...',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..OO.',
                     '..O..',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '...O.',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..O..',
                     '.OO..',
                     '.....']]

L_SHAPE_TEMPLATE = [['.....',
                     '...O.',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..O..',
                     '..OO.',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '.O...',
                     '.....'],
                    ['.....',
                     '.OO..',
                     '..O..',
                     '..O..',
                     '.....']]

T_SHAPE_TEMPLATE = [['.....',
                     '..O..',
                     '.OOO.',
                     '.....',
                     '.....'],
                    ['.....',
                     '..O..',
                     '..OO.',
                     '..O..',
                     '.....'],
                    ['.....',
                     '.....',
                     '.OOO.',
                     '..O..',
                     '.....'],
                    ['.....',
                     '..O..',
                     '.OO..',
                     '..O..',
                     '.....']]

PIECES = {'S': S_SHAPE_TEMPLATE,
          'Z': Z_SHAPE_TEMPLATE,
          'J': J_SHAPE_TEMPLATE,
          'L': L_SHAPE_TEMPLATE,
          'I': I_SHAPE_TEMPLATE,
          'O': O_SHAPE_TEMPLATE,
          'T': T_SHAPE_TEMPLATE}


class TetrisGame:
    def __init__(self):
        self.board = self.get_blank_board()
        self.score = 0
        self.level = 1
        self.falling_piece = None
        self.next_piece = self.get_new_piece()
        self.held_piece = None
        self.can_hold = True
        self.game_over = False
        self.fall_freq = 0.27
        self.initialize_metrics()

    def initialize_metrics(self):
        self.lines_cleared = 0
        self.total_pieces = 0
        self.max_height = 0
        self.holes = 0
        self.bumpiness = 0
        
    def get_blank_board(self):
        return [[BLANK for _ in range(BOARDHEIGHT)] for _ in range(BOARDWIDTH)]
    
    def get_new_piece(self):
        shape = random.choice(list(PIECES.keys()))
        self.total_pieces += 1
        return {
            'shape': shape,
            'rotation': random.randint(0, len(PIECES[shape]) - 1),
            'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
            'y': -2,
            'color': random.randint(1, len(COLORS))
        }
    
    def add_to_board(self, piece):
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if PIECES[piece['shape']][piece['rotation']][y][x] != BLANK and PIECES[piece['shape']][piece['rotation']][y][x] != '.':
                    self.board[x + piece['x']][y + piece['y']] = piece['color']
    
    def is_on_board(self, x, y):
        return 0 <= x < BOARDWIDTH and y < BOARDHEIGHT
    
    def is_valid_position(self, piece, adj_x=0, adj_y=0):
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                is_above_board = y + piece['y'] + adj_y < 0
                if is_above_board or PIECES[piece['shape']][piece['rotation']][y][x] == BLANK or PIECES[piece['shape']][piece['rotation']][y][x] == '.':
                    continue
                if not self.is_on_board(x + piece['x'] + adj_x, y + piece['y'] + adj_y):
                    return False
                if self.board[x + piece['x'] + adj_x][y + piece['y'] + adj_y] != BLANK:
                    return False
        return True
    
    def is_complete_line(self, y):
        return all(self.board[x][y] != BLANK for x in range(BOARDWIDTH))
    
    def remove_complete_lines(self):
        num_lines_removed = 0
        y = BOARDHEIGHT - 1
        while y >= 0:
            if self.is_complete_line(y):
                for pull_down_y in range(y, 0, -1):
                    for x in range(BOARDWIDTH):
                        self.board[x][pull_down_y] = self.board[x][pull_down_y-1]
                for x in range(BOARDWIDTH):
                    self.board[x][0] = BLANK
                num_lines_removed += 1
            else:
                y -= 1
        
        self.lines_cleared += num_lines_removed
        line_scores = [0, 40, 100, 300, 1200]
        if num_lines_removed > 0:
            self.score += line_scores[min(num_lines_removed, 4)] * self.level
            
        return num_lines_removed
    
    def calculate_level_and_fall_freq(self):
        self.level = min(int(self.lines_cleared / 10) + 1, 15)
        self.fall_freq = max(0.05, 0.27 - (self.level * 0.02))
        
    def hold_piece(self):
        if not self.can_hold:
            return
            
        if self.held_piece is None:
            self.held_piece = {
                'shape': self.falling_piece['shape'],
                'rotation': 0,
                'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                'y': -2,
                'color': self.falling_piece['color']
            }
            self.falling_piece = self.next_piece
            self.next_piece = self.get_new_piece()
        else:
            temp = self.held_piece
            self.held_piece = {
                'shape': self.falling_piece['shape'],
                'rotation': 0,
                'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                'y': -2,
                'color': self.falling_piece['color']
            }
            self.falling_piece = {
                'shape': temp['shape'],
                'rotation': 0,
                'x': int(BOARDWIDTH / 2) - int(TEMPLATEWIDTH / 2),
                'y': -2,
                'color': temp['color']
            }
            
        self.can_hold = False
        
    def get_landing_position(self, piece):
        landing_piece = piece.copy()
        for i in range(1, BOARDHEIGHT):
            if not self.is_valid_position(landing_piece, adj_y=i):
                break
        landing_piece['y'] += i - 1
        return landing_piece
    
    def get_board_state_features(self):
        heights = [0] * BOARDWIDTH
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                if self.board[x][y] != BLANK:
                    heights[x] = BOARDHEIGHT - y
                    break
        
        holes = 0
        for x in range(BOARDWIDTH):
            hole_found = False
            for y in range(BOARDHEIGHT):
                if self.board[x][y] != BLANK:
                    hole_found = True
                elif hole_found:
                    holes += 1
        
        bumpiness = 0
        for i in range(BOARDWIDTH - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        complete_lines = sum(1 for y in range(BOARDHEIGHT) if self.is_complete_line(y))
        
        aggregate_height = sum(heights)
        
        self.max_height = max(heights) if heights else 0
        self.holes = holes
        self.bumpiness = bumpiness
        
        return {
            'heights': heights,
            'max_height': max(heights) if heights else 0,
            'holes': holes,
            'bumpiness': bumpiness,
            'complete_lines': complete_lines,
            'aggregate_height': aggregate_height,
            'score': self.score,
            'level': self.level
        }


class TetrisRL:
    def __init__(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999, gamma=0.95, learning_rate=0.001):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.actions = ["LEFT", "RIGHT", "ROTATE", "DOWN", "DROP", "HOLD"]
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=6, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def get_state_vector(self, state_features):
        return np.array([
            state_features['max_height'] / BOARDHEIGHT,
            state_features['holes'] / (BOARDWIDTH * BOARDHEIGHT),
            state_features['bumpiness'] / (BOARDWIDTH * BOARDHEIGHT),
            state_features['complete_lines'] / BOARDHEIGHT,
            state_features['aggregate_height'] / (BOARDWIDTH * BOARDHEIGHT),
            state_features['score'] / 10000
        ])
    
    def get_action(self, state_features):
        state = self.get_state_vector(state_features)
        
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return self.actions[np.argmax(act_values[0])]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            self.get_state_vector(state),
            self.actions.index(action),
            reward,
            self.get_state_vector(next_state) if next_state else None,
            done
        ))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action_idx, reward, next_state, done in minibatch:
            target = reward
            if not done and next_state is not None:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action_idx] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath):
        self.model.save(filepath)
        
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


class TetrisRenderer:
    def __init__(self, game):
        self.game = game
        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.display_surf = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        self.basic_font = pygame.font.Font('freesansbold.ttf', 18)
        self.big_font = pygame.font.Font('freesansbold.ttf', 100)
        pygame.display.set_caption('Tetromino RL')
        
    def draw_box(self, x, y, color, pixel_x=None, pixel_y=None):
        if color == BLANK:
            return
            
        if pixel_x is None and pixel_y is None:
            pixel_x, pixel_y = self.convert_to_pixel_coords(x, y)
            
        pygame.draw.rect(self.display_surf, COLORS[color-1], (pixel_x + 1, pixel_y + 1, BOXSIZE - 1, BOXSIZE - 1))
        pygame.draw.rect(self.display_surf, LIGHTCOLORS[color-1], (pixel_x + 1, pixel_y + 1, BOXSIZE - 4, BOXSIZE - 4))
    
    def convert_to_pixel_coords(self, box_x, box_y):
        return (XMARGIN + (box_x * BOXSIZE)), (TOPMARGIN + (box_y * BOXSIZE))
    
    def draw_board(self):
        pygame.draw.rect(self.display_surf, BORDERCOLOR, 
                         (XMARGIN - 3, TOPMARGIN - 7, (BOARDWIDTH * BOXSIZE) + 8, (BOARDHEIGHT * BOXSIZE) + 8), 5)
        
        pygame.draw.rect(self.display_surf, BGCOLOR, 
                         (XMARGIN, TOPMARGIN, BOXSIZE * BOARDWIDTH, BOXSIZE * BOARDHEIGHT))
        
        for x in range(BOARDWIDTH):
            for y in range(BOARDHEIGHT):
                self.draw_box(x, y, self.game.board[x][y])
    
    def draw_piece(self, piece, pixel_x=None, pixel_y=None):
        shape_to_draw = PIECES[piece['shape']][piece['rotation']]
        if pixel_x is None and pixel_y is None:
            pixel_x, pixel_y = self.convert_to_pixel_coords(piece['x'], piece['y'])
            
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shape_to_draw[y][x] != BLANK and shape_to_draw[y][x] != '.':
                    self.draw_box(None, None, piece['color'], 
                                 pixel_x + (x * BOXSIZE), pixel_y + (y * BOXSIZE))
    
    def draw_ghost_piece(self, piece):
        landing_piece = self.game.get_landing_position(piece)
        
        ghost_surf = pygame.Surface((BOXSIZE, BOXSIZE), pygame.SRCALPHA)
        pygame.draw.rect(ghost_surf, (*COLORS[piece['color']-1], 128), 
                        (0, 0, BOXSIZE - 1, BOXSIZE - 1))
        
        shape_to_draw = PIECES[landing_piece['shape']][landing_piece['rotation']]
        pixel_x, pixel_y = self.convert_to_pixel_coords(landing_piece['x'], landing_piece['y'])
        
        for x in range(TEMPLATEWIDTH):
            for y in range(TEMPLATEHEIGHT):
                if shape_to_draw[y][x] != BLANK and shape_to_draw[y][x] != '.':
                    self.display_surf.blit(ghost_surf, 
                                         (pixel_x + (x * BOXSIZE), pixel_y + (y * BOXSIZE)))
    
    def draw_next_piece(self):
        next_surf = self.basic_font.render('Next:', True, TEXTCOLOR)
        next_rect = next_surf.get_rect()
        next_rect.topleft = (WINDOWWIDTH - 120, 80)
        self.display_surf.blit(next_surf, next_rect)
        
        self.draw_piece(self.game.next_piece, pixel_x=WINDOWWIDTH-120, pixel_y=100)
    
    def draw_held_piece(self):
        hold_surf = self.basic_font.render('Hold:', True, TEXTCOLOR)
        hold_rect = hold_surf.get_rect()
        hold_rect.topleft = (WINDOWWIDTH - 120, 180)
        self.display_surf.blit(hold_surf, hold_rect)
        
        if self.game.held_piece:
            self.draw_piece(self.game.held_piece, pixel_x=WINDOWWIDTH-120, pixel_y=200)
    
    def draw_status(self):
        score_surf = self.basic_font.render(f'Score: {self.game.score}', True, TEXTCOLOR)
        score_rect = score_surf.get_rect()
        score_rect.topleft = (WINDOWWIDTH - 150, 20)
        self.display_surf.blit(score_surf, score_rect)
        
        level_surf = self.basic_font.render(f'Level: {self.game.level}', True, TEXTCOLOR)
        level_rect = level_surf.get_rect()
        level_rect.topleft = (WINDOWWIDTH - 150, 50)
        self.display_surf.blit(level_surf, level_rect)
        
        lines_surf = self.basic_font.render(f'Lines: {self.game.lines_cleared}', True, TEXTCOLOR)
        lines_rect = lines_surf.get_rect()
        lines_rect.topleft = (WINDOWWIDTH - 150, 280)
        self.display_surf.blit(lines_surf, lines_rect)
        
        if hasattr(self, 'agent'):
            eps_surf = self.basic_font.render(f'Epsilon: {self.agent.epsilon:.2f}', True, TEXTCOLOR)
            eps_rect = eps_surf.get_rect()
            eps_rect.topleft = (WINDOWWIDTH - 150, 310)
            self.display_surf.blit(eps_surf, eps_rect)
    
    def show_text_screen(self, text):
        self.display_surf.fill(BGCOLOR)
        
        title_surf, title_rect = self.make_text_objs(text, self.big_font, TEXTSHADOWCOLOR)
        title_rect.center = (int(WINDOWWIDTH / 2), int(WINDOWHEIGHT / 2))
        self.display_surf.blit(title_surf, title_rect)
        
        title_surf, title_rect = self.make_text_objs(text, self.big_font, TEXTCOLOR)
        title_rect.center = (int(WINDOWWIDTH / 2) - 3, int(WINDOWHEIGHT / 2) - 3)
        self.display_surf.blit(title_surf, title_rect)
        
        press_key_surf, press_key_rect = self.make_text_objs('Press a key to play.', 
                                                            self.basic_font, TEXTCOLOR)
        press_key_rect.center = (int(WINDOWWIDTH / 2), int(WINDOWHEIGHT / 2) + 100)
        self.display_surf.blit(press_key_surf, press_key_rect)
        
        pygame.display.update()
        self.fps_clock.tick()
        
        while self.check_for_key_press() is None:
            pygame.display.update()
            self.fps_clock.tick()
    
    def make_text_objs(self, text, font, color):
        surf = font.render(text, True, color)
        return surf, surf.get_rect()
    
    def check_for_key_press(self):
        self.check_for_quit()
        
        for event in pygame.event.get([KEYDOWN, KEYUP]):
            if event.type == KEYDOWN:
                continue
            return event.key
        return None
    
    def check_for_quit(self):
        for event in pygame.event.get(QUIT):
            self.terminate()
        for event in pygame.event.get(KEYUP):
            if event.key == K_ESCAPE:
                self.terminate()
            pygame.event.post(event)
    
    def terminate(self):
        pygame.quit()
        sys.exit()


def run_game(use_ai=True, training_mode=True):
    game = TetrisGame()
    renderer = TetrisRenderer(game)
    
    if use_ai:
        agent = TetrisRL()
        renderer.agent = agent
        
        try:
            agent.load_model('tetris_model.h5')
            print("Loaded pre-trained model")
        except:
            print("No pre-trained model found, starting fresh")
    
    renderer.show_text_screen('Tetris RL')
    
    try:
        if random.randint(0, 1) == 0:
            pygame.mixer.music.load('assets/tetrisb.mid')
        else:
            pygame.mixer.music.load('assets/tetrisc.mid')
        pygame.mixer.music.play(-1, 0.0)
    except:
        pass
    
    last_move_down_time = time.time()
    last_move_sideways_time = time.time()
    last_fall_time = time.time()
    
    moving_down = False
    moving_left = False
    moving_right = False
    
    game.falling_piece = game.next_piece
    game.next_piece = game.get_new_piece()
    
    while True:
        if game.falling_piece is None:
            game.falling_piece = game.next_piece
            game.next_piece = game.get_new_piece()
            game.can_hold = True
            last_fall_time = time.time()
            
            if not game.is_valid_position(game.falling_piece):
                game.game_over = True
                break
        
        current_state = game.get_board_state_features()
        
        if use_ai:
            action = agent.get_action(current_state)
        else:
            action = None
            for event in pygame.event.get():
                if event.type == KEYUP:
                    if event.key == K_p:
                        renderer.display_surf.fill(BGCOLOR)
                        pygame.mixer.music.stop()
                        renderer.show_text_screen('Paused')
                        pygame.mixer.music.play(-1, 0.0)
                        last_fall_time = time.time()
                        last_move_down_time = time.time()
                        last_move_sideways_time = time.time()
                    elif event.key == K_LEFT:
                        moving_left = False
                    elif event.key == K_RIGHT:
                        moving_right = False
                    elif event.key == K_DOWN:
                        moving_down = False
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        moving_left = True
                        moving_right = False
                        action = "LEFT"
                    elif event.key == K_RIGHT:
                        moving_right = True
                        moving_left = False
                        action = "RIGHT"
                    elif event.key == K_UP:
                        action = "ROTATE"
                    elif event.key == K_DOWN:
                        moving_down = True
                        action = "DOWN"
                    elif event.key == K_SPACE:
                        action = "DROP"
                    elif event.key == K_c:
                        action = "HOLD"
        
        reward = 0
        if action == "LEFT" and game.is_valid_position(game.falling_piece, adj_x=-1):
            game.falling_piece['x'] -= 1
            moving_left = True
            moving_right = False
            last_move_sideways_time = time.time()
        elif action == "RIGHT" and game.is_valid_position(game.falling_piece, adj_x=1):
            game.falling_piece['x'] += 1
            moving_right = True
            moving_left = False
            last_move_sideways_time = time.time()
        elif action == "ROTATE":
            game.falling_piece['rotation'] = (game.falling_piece['rotation'] + 1) % len(PIECES[game.falling_piece['shape']])
            if not game.is_valid_position(game.falling_piece):
                game.falling_piece['rotation'] = (game.falling_piece['rotation'] - 1) % len(PIECES[game.falling_piece['shape']])
        elif action == "DOWN":
            if game.is_valid_position(game.falling_piece, adj_y=1):
                game.falling_piece['y'] += 1
                moving_down = True
                last_move_down_time = time.time()
                reward = 1  # Small reward for moving down
        elif action == "DROP":
            moving_down = False
            moving_left = False
            moving_right = False
            
            drop_distance = 0
            while game.is_valid_position(game.falling_piece, adj_y=drop_distance+1):
                drop_distance += 1
            
            game.falling_piece['y'] += drop_distance
            reward = drop_distance * 2  # Reward based on drop distance
            
            game.add_to_board(game.falling_piece)
            lines_cleared = game.remove_complete_lines()
            reward += lines_cleared * 100  # Big reward for clearing lines
            game.calculate_level_and_fall_freq()
            game.falling_piece = None
        elif action == "HOLD":
            game.hold_piece()
        
        if (moving_left or moving_right) and time.time() - last_move_sideways_time > MOVESIDEWAYSFREQ:
            if moving_left and game.is_valid_position(game.falling_piece, adj_x=-1):
                game.falling_piece['x'] -= 1
            elif moving_right and game.is_valid_position(game.falling_piece, adj_x=1):
                game.falling_piece['x'] += 1
            last_move_sideways_time = time.time()
        
        if moving_down and time.time() - last_move_down_time > MOVEDOWNFREQ and game.is_valid_position(game.falling_piece, adj_y=1):
            game.falling_piece['y'] += 1
            last_move_down_time = time.time()
        
        if time.time() - last_fall_time > game.fall_freq:
            if not game.is_valid_position(game.falling_piece, adj_y=1):
                game.add_to_board(game.falling_piece)
                lines_cleared = game.remove_complete_lines()
                reward += lines_cleared * 100
                game.calculate_level_and_fall_freq()
                game.falling_piece = None
            else:
                game.falling_piece['y'] += 1
                last_fall_time = time.time()
        
        next_state = game.get_board_state_features()
        
        if use_ai and training_mode and action:
            agent.remember(current_state, action, reward, next_state, game.falling_piece is None)
            agent.replay()
        
        renderer.display_surf.fill(BGCOLOR)
        renderer.draw_board()
        renderer.draw_status()
        renderer.draw_next_piece()
        renderer.draw_held_piece()
        
        if game.falling_piece:
            renderer.draw_ghost_piece(game.falling_piece)
            renderer.draw_piece(game.falling_piece)
        
        pygame.display.update()
        renderer.fps_clock.tick(FPS)
    
    if use_ai and training_mode:
        agent.save_model('tetris_model.h5')
    
    pygame.mixer.music.stop()
    renderer.show_text_screen('Game Over')


def main():
    run_game(use_ai=True, training_mode=True)


if __name__ == '__main__':
    main()
