import 'package:bloc_test/bloc_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:fpdart/fpdart.dart';
import 'package:neurobridge_mobile/features/auth/domain/entities/user.dart';
import 'package:neurobridge_mobile/features/auth/domain/repositories/auth_repository.dart';
import 'package:neurobridge_mobile/features/auth/presentation/bloc/auth_bloc.dart';
import 'package:neurobridge_mobile/features/auth/presentation/bloc/auth_event.dart';
import 'package:neurobridge_mobile/features/auth/presentation/bloc/auth_state.dart';

class MockAuthRepository extends Mock implements AuthRepository {}

void main() {
  late AuthBloc authBloc;
  late MockAuthRepository mockAuthRepository;

  setUp(() {
    mockAuthRepository = MockAuthRepository();
    authBloc = AuthBloc(authRepository: mockAuthRepository);
  });

  const tUser = User(id: '1', email: 'test@example.com', name: 'Test User');

  group('AuthCheckRequested', () {
    blocTest<AuthBloc, AuthState>(
      'emits [Authenticated] when repository returns user',
      build: () {
        when(() => mockAuthRepository.getCurrentUser())
            .thenAnswer((_) async => const Right(tUser));
        return authBloc;
      },
      act: (bloc) => bloc.add(AuthCheckRequested()),
      expect: () => [
        const Authenticated(user: tUser),
      ],
    );

    blocTest<AuthBloc, AuthState>(
      'emits [Unauthenticated] when repository returns failure',
      build: () {
        when(() => mockAuthRepository.getCurrentUser())
            .thenAnswer((_) async => const Left(ServerFailure('Error')));
        return authBloc;
      },
      act: (bloc) => bloc.add(AuthCheckRequested()),
      expect: () => [
        Unauthenticated(),
      ],
    );
  });
}
